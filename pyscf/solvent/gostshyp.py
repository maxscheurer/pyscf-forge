# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GOSTSHYP pressure solvation model.

The GOSTSHYP (Gaussian On Surface Tesserae Simulate HYdrostatic Pressure)
model applies an isotropic pressure to a molecular cavity surface using
Gaussian-weighted integrals.

Supports two cavity types:
  - 'vdw': van der Waals surface with scaled Bondi radii
  - 'vdw/occ': Occluded van der Waals surface (crevice-free)

References:
    J. Chem. Theory Comput. 2021, 17, 1, 583-597
    https://doi.org/10.1021/acs.jctc.0c01212
"""

import numpy as np
from functools import cached_property

from pyscf import gto, lib, scf
from pyscf.lib import logger
from pyscf.solvent import _attach_solvent
from pyscf.solvent.pcm import gen_surface, modified_Bondi
from pyscf.solvent.grad.pcm import get_dF_dA

# Pressure conversion: 1 MPa = 3.3989309735473356e-08 Hartree/Bohr^3
MPA_TO_AU = 3.3989309735473356e-08


def fakemol_for_gaussian(coords, exponents, l=0, cart=True, coeffs=None):
    """Build a fake Mole object representing auxiliary Gaussians.

    Parameters
    ----------
    coords : ndarray of shape (n, 3)
        Gaussian center coordinates in Bohr.
    exponents : ndarray of shape (n,)
        Gaussian exponents.
    l : int
        Angular momentum (0=s, 1=p, 2=d, 3=f).
    cart : bool
        Use Cartesian Gaussians.
    coeffs : ndarray of shape (n,), optional
        Contraction coefficients (default: ones).

    Returns
    -------
    fakemol : gto.Mole
    """
    nbas = coords.shape[0]
    if coeffs is None:
        coeffs = np.ones_like(exponents)
    angmom = np.full(nbas, l, dtype=np.int32)

    ang_norm = {0: 2.0 * np.sqrt(np.pi),
                1: 2.0 * np.sqrt(np.pi / 3),
                2: 1.0,
                3: 1.0}

    fakeatm = np.zeros((nbas, gto.mole.ATM_SLOTS), dtype=np.int32)
    fakebas = np.zeros((nbas, gto.mole.BAS_SLOTS), dtype=np.int32)
    fakeenv = [0] * gto.mole.PTR_ENV_START
    ptr = gto.mole.PTR_ENV_START

    fakeatm[:, gto.mole.PTR_COORD] = np.arange(ptr, ptr + nbas * 3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas * 3

    fakebas[:, gto.mole.ATOM_OF] = np.arange(nbas)
    fakebas[:, gto.mole.ANG_OF] = angmom
    fakebas[:, gto.mole.NPRIM_OF] = 1
    fakebas[:, gto.mole.NCTR_OF] = 1
    fakebas[:, gto.mole.PTR_EXP] = ptr + np.arange(nbas) * 2
    fakebas[:, gto.mole.PTR_COEFF] = ptr + np.arange(nbas) * 2 + 1

    coeff = ang_norm[l] * coeffs
    fakeenv.append(np.vstack((exponents, coeff)).T.ravel())

    fakemol = gto.Mole()
    fakemol.cart = cart
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = np.hstack(fakeenv)
    fakemol._built = True
    return fakemol


def compute_surface_normals(atom_coords, grid, atom_idx):
    """Compute inward-pointing unit normals for surface grid points.

    Parameters
    ----------
    atom_coords : ndarray of shape (natm, 3)
    grid : ndarray of shape (ngrids, 3)
    atom_idx : ndarray of shape (ngrids,)

    Returns
    -------
    normals : ndarray of shape (ngrids, 3)
    """
    ref_coords = atom_coords[atom_idx]
    dr = ref_coords - grid
    dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
    return dr / dr_norm


class GOSTSHYP(lib.StreamObject):
    """
    GOSTSHYP pressure solvation model.

    Attributes
    ----------
    mol : pyscf.gto.Mole
        Molecular object.
    pressure_mpa : float
        Applied pressure in MPa (default: 50000 = 50 GPa).
    npoints : int
        Number of Lebedev grid points per atom (default: 110).
    scaling_factor : float
        Van der Waals radii scaling factor (default: 1.2).
    cavity : str
        Cavity type: 'vdw' or 'vdw/occ' (default: 'vdw/occ').
    r_ext : float
        Extension radius for vdW/OCC in Bohr (default: 0.4724 ~ 0.25 Ang).
    direct : bool
        If True, compute integrals on-the-fly without caching (default: False).
    """

    def __init__(self, mol, options=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        if options is None:
            options = {}
        self.pressure_mpa = options.get('pressure_mpa', 50_000)
        self.npoints = options.get('npoints', 110)
        self.scaling_factor = options.get('scaling_factor', 1.2)
        self.cavity = options.get('cavity', 'vdw/occ')
        self.r_ext = options.get('r_ext', 0.4724)  # Bohr (0.25 Ang)
        self.direct = options.get('direct', False)

        self.frozen = False
        self.equilibrium_solvation = False
        self.e = None
        self.v = None
        self.amplitudes = None

        self.build()

    @property
    def pressure_au(self):
        """Pressure in atomic units (Hartree/Bohr^3)."""
        return self.pressure_mpa * MPA_TO_AU

    def build(self, mol=None):
        """Build surface tessellation."""
        if mol is not None:
            self.mol = mol
        mol = self.mol

        rad = self.scaling_factor * modified_Bondi

        if self.cavity == 'vdw/occ':
            r_ext = self.r_ext
            rad_outer = rad + r_ext
            self.surface_dict = gen_surface(mol, ng=self.npoints, rad=rad_outer)

            norm_vec = self.surface_dict['norm_vec']
            grid_outer = self.surface_dict['grid_coords']
            grid_inner = grid_outer - r_ext * norm_vec

            R_outer = self.surface_dict['R_vdw']
            R_inner = R_outer - r_ext
            ratio_sq = (R_inner / R_outer) ** 2
            area_occ = self.surface_dict['area'] * ratio_sq

            self.surface_dict['grid_coords_outer'] = grid_outer
            self.surface_dict['grid_coords'] = grid_inner
            self.surface_dict['area'] = area_occ
            self.surface_dict['R_vdw'] = R_inner
        else:
            self.surface_dict = gen_surface(mol, ng=self.npoints, rad=rad)

        # Build atom index
        atom_idx = np.zeros(len(self.surface_dict['area']), dtype=np.int32)
        for i, (start, stop) in enumerate(self.surface_dict['gslice_by_atom']):
            atom_idx[start:stop] = i

        self.grid_coords = self.surface_dict['grid_coords']
        self.areas = self.surface_dict['area']
        self.atom_idx = atom_idx
        self.widths = np.pi * np.log(2) / self.areas
        self.n_gaussian = len(self.areas)
        self.surface_normals = compute_surface_normals(
            mol.atom_coords(), self.grid_coords, self.atom_idx)

        # Clear cached properties
        self.__dict__.pop('gtilde', None)
        self.__dict__.pop('force_operators', None)

        logger.info(self, 'GOSTSHYP: %d surface Gaussians (cavity=%s)',
                    self.n_gaussian, self.cavity)
        return self

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'pressure = %.1f MPa (%.6e a.u.)',
                    self.pressure_mpa, self.pressure_au)
        logger.info(self, 'npoints = %d', self.npoints)
        logger.info(self, 'scaling_factor = %.2f', self.scaling_factor)
        logger.info(self, 'cavity = %s', self.cavity)
        if self.cavity == 'vdw/occ':
            logger.info(self, 'r_ext = %.4f Bohr', self.r_ext)
        logger.info(self, 'direct = %s', self.direct)
        logger.info(self, 'n_gaussian = %d', self.n_gaussian)
        return self

    def check_sanity(self):
        return self

    def kernel(self, dm):
        """Compute GOSTSHYP energy and Fock matrix contribution.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix.

        Returns
        -------
        energy : float
        fock : ndarray of shape (nao, nao)
        """
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        if self.direct:
            return self._kernel_direct(dm)
        else:
            return self._kernel_cached(dm)

    def _kernel_cached(self, dm):
        """Cached mode: uses precomputed gtilde and force_operators."""
        forces = np.einsum('bkg,bk->g', self.force_operators, dm, optimize=True)
        amplitudes = self.pressure_au * self.areas / forces
        self.amplitudes = amplitudes

        gtilde_expval = np.einsum('bkg,bk->g', self.gtilde, dm, optimize=True)

        fock1 = np.einsum('g,bkg->bk', amplitudes, self.gtilde, optimize=True)
        fock2 = -self.pressure_au * self.areas * gtilde_expval / (forces ** 2)
        fock2 = np.einsum('g,bkg->bk', fock2, self.force_operators, optimize=True)

        energy = np.vdot(fock1, dm)
        fock = fock1 + fock2

        self.e = energy
        self.v = fock
        logger.info(self, 'GOSTSHYP energy: %.10f', energy)
        return energy, fock

    def _kernel_direct(self, dm):
        """Integral-direct mode: compute integrals on-the-fly in chunks."""
        mol = self.mol
        nao = mol.nao_nr()

        max_memreq = 5 * self.n_gaussian * nao**2 * 8.0 / 1e6
        max_memory = max(2000, mol.max_memory * 0.9 - lib.current_memory()[0])
        n_chunks = max(1, int(max_memreq // max_memory + 1))

        shells = np.arange(self.n_gaussian)
        chunks = np.array_split(shells, n_chunks)

        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        gmol_p = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=2.0 * self.widths)
        supermol = mol + gmol
        supermol_p = mol + gmol_p

        energy = 0.0
        fock = np.zeros_like(dm)
        self.amplitudes = np.zeros(self.n_gaussian)

        for shell_slice in chunks:
            off1 = int(shell_slice[0])
            off2 = len(shell_slice)
            slices = (0, mol.nbas, 0, mol.nbas,
                      mol.nbas + off1, mol.nbas + off1 + off2)

            overlap3_s = supermol.intor('int3c1e', shls_slice=slices, aosym='s1')
            overlap3_p = supermol_p.intor(
                'int3c1e', shls_slice=slices, aosym='s1'
            ).reshape(nao, nao, -1, 3)

            force_ops = np.einsum(
                'bkgc,gc->bkg', overlap3_p,
                self.surface_normals[shell_slice], optimize=True)
            forces = np.einsum('bk,bkg->g', dm, force_ops, optimize=True)
            amplitudes = self.pressure_au * self.areas[shell_slice] / forces
            self.amplitudes[shell_slice] = amplitudes

            f1 = np.einsum('g,bkg->bk', amplitudes, overlap3_s, optimize=True)

            gtilde_expval = np.einsum('bk,bkg->g', dm, overlap3_s, optimize=True)
            f2 = -self.pressure_au * self.areas[shell_slice] * gtilde_expval / (forces ** 2)
            f2 = np.einsum('g,bkg->bk', f2, force_ops, optimize=True)

            fock += f1 + f2
            energy += np.vdot(f1, dm)

        self.e = energy
        self.v = fock
        logger.info(self, 'GOSTSHYP energy: %.10f', energy)
        return energy, fock

    @cached_property
    def gtilde(self):
        """s-type 3-center overlap integrals [nao, nao, n_gaussian]."""
        mol = self.mol
        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        supermol = mol + gmol
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)
        ret = supermol.intor('int3c1e', shls_slice=slices, aosym='s1')
        return ret

    @cached_property
    def force_operators(self):
        """p-type 3-center overlaps contracted with normals [nao, nao, n_gaussian]."""
        mol = self.mol
        nao = mol.nao_nr()
        gmol_p = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=2.0 * self.widths)
        supermol = mol + gmol_p
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)
        overlap3 = supermol.intor(
            'int3c1e', shls_slice=slices, aosym='s1'
        ).reshape(nao, nao, -1, 3)
        force_ops = np.einsum(
            'bkgc,gc->bkg', overlap3, self.surface_normals, optimize=True)
        return force_ops

    def grad(self, dm):
        """Compute analytical nuclear gradient of GOSTSHYP energy.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix.

        Returns
        -------
        grad : ndarray of shape (natm, 3)
        """
        mol = self.mol
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        _, dareas = get_dF_dA(self.surface_dict)
        dareas = dareas.transpose(1, 2, 0)  # (natm, 3, ngrids)

        forces = np.einsum('bkg,bk->g', self.force_operators, dm, optimize=True)
        gtilde_expval = np.einsum('bkg,bk->g', self.gtilde, dm, optimize=True)

        # Term 1: area derivative
        dE1 = self.pressure_au * np.einsum(
            'acg,g->ac', dareas, gtilde_expval / forces, optimize=True)

        # Term 2: gtilde operator derivative
        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        supermol = mol + gmol
        slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)

        dPQ = supermol.intor('int3c1e_ip1', shls_slice=slices)
        dPQ = np.einsum('xijn,n->xij', dPQ, self.amplitudes, optimize=True)

        slices_g = (mol.nbas, mol.nbas + gmol.nbas, 0, mol.nbas, 0, mol.nbas)
        dG = supermol.intor('int3c1e_ip1', shls_slice=slices_g)

        aoslice = mol.aoslice_by_atom()
        nao = mol.nao_nr()
        dgtilde_braket = np.einsum('xij,ij->ix', dPQ, dm, optimize=True)
        dgtilde_braket += np.einsum('xij,ji->ix', dPQ, dm, optimize=True)

        dgtilde_gaussian = np.einsum(
            'xnij,n,ij->nx', dG, self.amplitudes, dm, optimize=True)
        gtilde_operator_grad = np.asarray(
            [np.sum(dgtilde_braket[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
        np.add.at(gtilde_operator_grad, self.atom_idx, dgtilde_gaussian)
        gtilde_operator_grad *= -1.0

        # d-orbital width gradient
        wgrad_prefs = -np.pi * np.log(2) / (self.areas ** 2)
        gmol_d = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=2,
            coeffs=wgrad_prefs * self.amplitudes)
        supermol_d = mol + gmol_d
        supermol_d.cart = True
        slices_d = (0, mol.nbas, 0, mol.nbas,
                    mol.nbas, mol.nbas + gmol_d.nbas)

        nao_cart = mol.nao_nr(cart=True)
        overlap3d = supermol_d.intor(
            'int3c1e', shls_slice=slices_d
        ).reshape(nao_cart, nao_cart, -1, 6)

        if not mol.cart:
            c2s = mol.cart2sph_coeff(normalized='sp')
            overlap3d = np.einsum(
                'ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s, optimize=True)

        diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
        imd = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
        dE_d = -np.einsum('acg,g->ac', dareas, imd, optimize=True)
        dE2 = gtilde_operator_grad + dE_d

        # Term 3: force operator derivative
        coeffs_fop = (
            -2.0 * self.pressure_au * self.areas * gtilde_expval
            * self.widths / (forces * forces))
        gmol_f = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=coeffs_fop)
        supermol_f = mol + gmol_f
        slices_f = (0, mol.nbas, 0, mol.nbas,
                    mol.nbas, mol.nbas + gmol_f.nbas)
        dpq = supermol_f.intor(
            'int3c1e_ip1', shls_slice=slices_f
        ).reshape(3, nao, nao, -1, 3)
        dpq *= self.surface_normals

        slices_fg = (mol.nbas, mol.nbas + gmol_f.nbas, 0, mol.nbas, 0, mol.nbas)
        dG_f = supermol_f.intor(
            'int3c1e_ip1', shls_slice=slices_fg
        ).reshape(3, -1, 3, nao, nao)

        dpq_ix = np.einsum('xijnp,ij->ix', dpq, dm, optimize=True)
        dpq_ix += np.einsum('xijnp,ji->ix', dpq, dm, optimize=True)

        dG_f = np.einsum('xnpij,np->xnij', dG_f, self.surface_normals, optimize=True)
        dG_f = np.einsum('xnij,ij->nx', dG_f, dm, optimize=True)

        force_operator_grad = np.asarray(
            [np.sum(dpq_ix[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
        np.add.at(force_operator_grad, self.atom_idx, dG_f)
        force_operator_grad *= -1.0

        # f-orbital width gradient for force operators
        coeffs_f2 = -2.0 * self.widths * wgrad_prefs
        gmol_ft = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=3, coeffs=coeffs_f2)
        supermol_ft = mol + gmol_ft
        supermol_ft.cart = True
        slices_ft = (0, mol.nbas, 0, mol.nbas,
                     mol.nbas, mol.nbas + gmol_ft.nbas)
        overlap3f = supermol_ft.intor(
            'int3c1e', shls_slice=slices_ft
        ).reshape(nao_cart, nao_cart, -1, 10)

        if not mol.cart:
            c2s = mol.cart2sph_coeff(normalized='sp')
            overlap3f = np.einsum(
                'ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s, optimize=True)

        # Trace over Cartesian components -> x, y, z contributions
        xf = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
        yf = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
        zf = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
        dx = np.einsum('ijg,ij->g', xf, dm, optimize=True)
        dy = np.einsum('ijg,ij->g', yf, dm, optimize=True)
        dz = np.einsum('ijg,ij->g', zf, dm, optimize=True)
        dr = np.vstack((dx, dy, dz)).T

        rf2 = 1.0 / (forces * forces)
        dr *= self.surface_normals
        dFdR = dareas * wgrad_prefs * forces / self.widths
        dFdR += np.einsum('gc,axg->axg', dr, dareas, optimize=True)
        width_grad_ftype = (
            -self.pressure_au * np.einsum(
                'g,g,axg,g->ax', self.areas, gtilde_expval,
                dFdR, rf2, optimize=True))

        dE3 = force_operator_grad + width_grad_ftype

        return dE1 + dE2 + dE3

    def reset(self, mol=None):
        """Reset for geometry optimization / scanner."""
        if mol is not None:
            self.mol = mol
            self.__dict__.pop('gtilde', None)
            self.__dict__.pop('force_operators', None)
            self.build()
        return self

    def nuc_grad_method(self):
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)


@lib.with_doc(_attach_solvent._for_scf.__doc__)
def gostshyp_for_scf(mf, solvent_obj=None, dm=None):
    """Attach GOSTSHYP solvent model to SCF method."""
    if solvent_obj is None:
        solvent_obj = GOSTSHYP(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)


# Inject GOSTSHYP into SCF classes
scf.hf.SCF.GOSTSHYP = gostshyp_for_scf
