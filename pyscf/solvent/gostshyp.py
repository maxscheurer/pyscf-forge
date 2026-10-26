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

    J. Chem. Theory Comput. 2025, 21, 2, 764-776
    https://doi.org/10.1021/acs.jctc.4c01502
"""

import numpy as np
from functools import cached_property

from pyscf import gto, lib, scf
from pyscf.gto import moleintor
from pyscf.lib import logger
from pyscf.solvent import _attach_solvent
from pyscf.solvent.pcm import gen_surface, modified_Bondi

# Pressure conversion: 1 MPa = 3.3989309735473356e-08 Hartree/Bohr^3
MPA_TO_AU = 3.3989309735473356e-08

# libcint >= 6.2.0 provides the int3c1e second derivatives needed by the
# Hessian, but PySCF does not register them in its intor table (yet).
moleintor._INTOR_FUNCTIONS.update({
    'int3c1e_ipip1': (9, 9),
    'int3c1e_ip1ip2': (9, 9),
    'int3c1e_ipvip1': (9, 9),
    'int3c1e_ipip2': (9, 9),
})


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
                3: 1.0,
                4: 1.0,
                5: 1.0}

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
    if np.any(dr_norm < 1e-14):
        raise ValueError(
            'Grid point coincides with its atom center; '
            'surface tessellation is degenerate.')
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
        If True, compute integrals on-the-fly without caching (default: True).
    area_thresh : float or None
        Grid points with area below this threshold (Bohr²) are pruned from
        the surface.  This removes numerically problematic small-area points
        that cause ill-conditioning in the Hessian (width ∝ 1/area²).
        Set to 0 or None to disable pruning (default: 1e-3).
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
        self.direct = options.get('direct', True)
        self.area_thresh = options.get('area_thresh', 1e-3)

        self.frozen = False
        self.equilibrium_solvation = False
        self.e = None
        self.v = None
        self.amplitudes = None
        self.forces = None
        self.gtilde_expval = None

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

            # Save outer surface_dict for gradient computation
            self._outer_surface_dict = dict(self.surface_dict)

            norm_vec = self.surface_dict['norm_vec']
            grid_outer = self.surface_dict['grid_coords']
            grid_inner = grid_outer - r_ext * norm_vec

            R_outer = self.surface_dict['R_vdw']
            R_inner = R_outer - r_ext
            ratio_sq = (R_inner / R_outer) ** 2
            self._occ_ratio_sq = ratio_sq
            area_occ = self.surface_dict['area'] * ratio_sq

            self.surface_dict['grid_coords_outer'] = grid_outer
            self.surface_dict['grid_coords'] = grid_inner
            self.surface_dict['area'] = area_occ
            self.surface_dict['R_vdw'] = R_inner
        else:
            self.surface_dict = gen_surface(mol, ng=self.npoints, rad=rad)
            self._outer_surface_dict = None
            self._occ_ratio_sq = None

        # Build atom index
        atom_idx = np.zeros(len(self.surface_dict['area']), dtype=np.int32)
        for i, (start, stop) in enumerate(self.surface_dict['gslice_by_atom']):
            atom_idx[start:stop] = i

        self.grid_coords = self.surface_dict['grid_coords']
        self.areas = self.surface_dict['area']
        if np.any(self.areas <= 0):
            raise ValueError(
                'Non-positive surface areas detected; cavity is degenerate.')
        self.atom_idx = atom_idx

        # Prune small-area points before computing derived quantities
        self._prune_small_areas()

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

    def _prune_small_areas(self):
        """Remove grid points with area below area_thresh from the surface.

        This eliminates numerically problematic points where the Gaussian
        width (∝ 1/area) becomes excessively large, causing ill-conditioning
        in derivative computations.  The removed points carry negligible
        surface area and energy (<µHa for typical thresholds).
        """
        if not self.area_thresh:
            return

        keep = self.areas >= self.area_thresh
        n_removed = int(np.sum(~keep))
        if n_removed == 0:
            return

        total_area = np.sum(self.areas)
        area_removed_frac = np.sum(self.areas[~keep]) / total_area

        if area_removed_frac > 0.01:
            logger.warn(self,
                'GOSTSHYP: area pruning removed %.1f%% of total surface area '
                '(%d/%d points, thresh=%.1e). Consider reducing area_thresh '
                'or inspecting the cavity.',
                100.0 * area_removed_frac, n_removed, len(self.areas),
                self.area_thresh)

        logger.info(self, 'GOSTSHYP: pruned %d/%d points (%.2e%% of area) '
                    'with area < %.1e',
                    n_removed, len(self.areas),
                    100.0 * area_removed_frac, self.area_thresh)

        # Apply mask to surface arrays
        self.grid_coords = self.grid_coords[keep]
        self.areas = self.areas[keep]
        self.atom_idx = self.atom_idx[keep]

        # Update surface_dict entries used by gradient/Hessian
        self.surface_dict['grid_coords'] = self.grid_coords
        self.surface_dict['area'] = self.areas
        if 'grid_coords_outer' in self.surface_dict:
            self.surface_dict['grid_coords_outer'] = (
                self.surface_dict['grid_coords_outer'][keep])
        if 'norm_vec' in self.surface_dict:
            self.surface_dict['norm_vec'] = self.surface_dict['norm_vec'][keep]
        if 'R_vdw' in self.surface_dict:
            self.surface_dict['R_vdw'] = self.surface_dict['R_vdw'][keep]

        # Rebuild gslice_by_atom from filtered atom_idx
        natm = self.mol.natm
        new_slices = []
        pos = 0
        for i in range(natm):
            count = int(np.sum(self.atom_idx == i))
            new_slices.append((pos, pos + count))
            pos += count
        self.surface_dict['gslice_by_atom'] = new_slices

        # Update occ-specific arrays if present
        if self._occ_ratio_sq is not None:
            self._occ_ratio_sq = self._occ_ratio_sq[keep]

        # Prune _outer_surface_dict (used in gradient for vdw/occ)
        if self._outer_surface_dict is not None:
            self._outer_surface_dict['grid_coords'] = (
                self._outer_surface_dict['grid_coords'][keep])
            self._outer_surface_dict['area'] = (
                self._outer_surface_dict['area'][keep])
            if 'norm_vec' in self._outer_surface_dict:
                self._outer_surface_dict['norm_vec'] = (
                    self._outer_surface_dict['norm_vec'][keep])
            if 'R_vdw' in self._outer_surface_dict:
                self._outer_surface_dict['R_vdw'] = (
                    self._outer_surface_dict['R_vdw'][keep])
            self._outer_surface_dict['gslice_by_atom'] = new_slices

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
        logger.info(self, 'area_thresh = %s', self.area_thresh)
        logger.info(self, 'n_gaussian = %d', self.n_gaussian)
        return self

    def check_sanity(self):
        if self.pressure_mpa <= 0:
            raise ValueError(f'pressure_mpa must be positive, got {self.pressure_mpa}')
        if self.scaling_factor <= 0:
            raise ValueError(f'scaling_factor must be positive, got {self.scaling_factor}')
        if self.cavity not in ('vdw', 'vdw/occ'):
            raise ValueError(f"cavity must be 'vdw' or 'vdw/occ', got '{self.cavity}'")
        if self.cavity == 'vdw/occ' and self.r_ext <= 0:
            raise ValueError(f'r_ext must be positive for vdw/occ cavity, got {self.r_ext}')
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
        nao = self.mol.nao_nr()
        dm_flat = dm.ravel(order='F')
        forces = dm_flat @ self.force_operators.reshape(nao * nao, -1, order='F')
        amplitudes = self.pressure_au * self.areas / forces
        mask = amplitudes < 0
        if np.any(mask):
            n_neg = np.count_nonzero(mask)
            logger.warn(self, 'GOSTSHYP: %d/%d (%.1f%%) negative amplitudes removed',
                        n_neg, amplitudes.size, 100.0 * n_neg / amplitudes.size)
            amplitudes[mask] = 0.0
            forces[mask] = 1.0  # avoid division by zero in gradient
        self.amplitudes = amplitudes
        self.forces = forces

        gtilde_expval = dm_flat @ self.gtilde.reshape(nao * nao, -1, order='F')
        self.gtilde_expval = gtilde_expval

        fock1 = (self.gtilde.reshape(nao * nao, -1, order='F') @ amplitudes
                 ).reshape(nao, nao, order='F')
        fock2 = -self.pressure_au * self.areas * gtilde_expval / (forces ** 2)
        fock2 = (self.force_operators.reshape(nao * nao, -1, order='F') @ fock2
                 ).reshape(nao, nao, order='F')

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
        self.forces = np.zeros(self.n_gaussian)
        self.gtilde_expval = np.zeros(self.n_gaussian)

        dm_flat = dm.ravel(order='F')
        nao2 = nao * nao

        for shell_slice in chunks:
            off1 = int(shell_slice[0])
            off2 = len(shell_slice)
            slices = (0, mol.nbas, 0, mol.nbas,
                      mol.nbas + off1, mol.nbas + off1 + off2)

            overlap3_s = supermol.intor('int3c1e', shls_slice=slices, aosym='s1')
            overlap3_p = supermol_p.intor(
                'int3c1e', shls_slice=slices, aosym='s1'
            ).reshape(nao, nao, -1, 3)

            normals = self.surface_normals[shell_slice]
            force_ops = np.einsum('bkgc,gc->bkg', overlap3_p, normals,
                                  optimize=False)
            force_ops_2d = force_ops.reshape(nao2, -1, order='F')
            forces = dm_flat @ force_ops_2d
            amplitudes = self.pressure_au * self.areas[shell_slice] / forces
            mask = amplitudes < 0
            if np.any(mask):
                n_neg = np.count_nonzero(mask)
                logger.warn(self, 'GOSTSHYP: %d/%d (%.1f%%) negative amplitudes removed',
                            n_neg, amplitudes.size, 100.0 * n_neg / amplitudes.size)
                amplitudes[mask] = 0.0
                forces[mask] = 1.0  # avoid division by zero in gradient
            self.amplitudes[shell_slice] = amplitudes
            self.forces[shell_slice] = forces

            overlap3_s_2d = overlap3_s.reshape(nao2, -1, order='F')
            f1 = (overlap3_s_2d @ amplitudes).reshape(nao, nao, order='F')

            gtilde_expval = dm_flat @ overlap3_s_2d
            self.gtilde_expval[shell_slice] = gtilde_expval
            f2 = -self.pressure_au * self.areas[shell_slice] * gtilde_expval / (forces ** 2)
            f2 = (force_ops_2d @ f2).reshape(nao, nao, order='F')

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
        force_ops = np.einsum('bkgc,gc->bkg', overlap3, self.surface_normals,
                              optimize=False)
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
        if self.forces is None:
            raise RuntimeError(
                'kernel() must be called before grad(). '
                'Forces have not been computed.')

        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        from pyscf.solvent._gostshyp_hess import grad
        return grad(self, dm)

    def hess_fd(self, dm, step=1e-4):
        """Semi-numerical Hessian via central finite differences of grad(dm).

        For each atom B, direction y, displaces the geometry +/- step,
        rebuilds the surface, recomputes intermediates with the SAME dm,
        and finite-differences the gradient.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix (held fixed).
        step : float
            Finite-difference step size in Bohr.

        Returns
        -------
        hess : ndarray of shape (natm, natm, 3, 3)
            d²E/dR_A dR_B at fixed density.
        """
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        mol = self.mol
        natm = mol.natm
        coords0 = mol.atom_coords().copy()
        hess = np.zeros((natm, natm, 3, 3))

        options = {
            'cavity': self.cavity,
            'pressure_mpa': self.pressure_mpa,
            'npoints': self.npoints,
            'scaling_factor': self.scaling_factor,
            'area_thresh': self.area_thresh,
        }
        if self.cavity == 'vdw/occ':
            options['r_ext'] = self.r_ext

        for B in range(natm):
            for y in range(3):
                # Plus displacement
                coords_p = coords0.copy()
                coords_p[B, y] += step
                mol_p = mol.copy()
                mol_p.set_geom_(coords_p, unit='Bohr')
                gost_p = GOSTSHYP(mol_p, options=options)
                gost_p.kernel(dm)
                grad_p = gost_p.grad(dm)

                # Minus displacement
                coords_m = coords0.copy()
                coords_m[B, y] -= step
                mol_m = mol.copy()
                mol_m.set_geom_(coords_m, unit='Bohr')
                gost_m = GOSTSHYP(mol_m, options=options)
                gost_m.kernel(dm)
                grad_m = gost_m.grad(dm)

                hess[:, B, :, y] = (grad_p - grad_m) / (2.0 * step)

        return hess

    def hess(self, dm):
        """Compute analytical GOSTSHYP contribution to the nuclear Hessian.

        Computes d²E_GOSTSHYP/dR_A dR_B at fixed density matrix. This is
        the explicit geometry contribution to the nuclear Hessian (the
        direct term, without orbital response).

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix.

        Returns
        -------
        de_solvent : ndarray of shape (natm, natm, 3, 3)
            Second derivatives of the GOSTSHYP energy.
        """
        if self.forces is None:
            raise RuntimeError(
                'kernel() must be called before hess(). '
                'Forces have not been computed.')

        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        from pyscf.solvent._gostshyp_hess import kernel as _hess_kernel
        return _hess_kernel(self, dm)

    def _B_dot_x(self, dm1):
        """Linear response of the GOSTSHYP Fock operator to density perturbations.

        Computes the matrix-vector product B·x where B is the 4-index kernel
        ∂V_μν/∂D_κλ and x = dm1_κλ. This is needed for the CPHF equations
        in the analytical Hessian.

        The GOSTSHYP Fock matrix is:
            V_μν(D) = Σ_g [α_g(D)·G̃_{g,μν} + β_g(D)·F̂_{g,μν}]

        where α_g = P·A_g/F_g and β_g = -P·A_g·e_g/F_g².

        The linear response to a density perturbation δD is:
            δV_μν = Σ_g [δα_g·G̃_{g,μν} + δβ_g·F̂_{g,μν}]

        with:
            δe_g = Tr[δD · G̃_g]
            δF_g = Tr[δD · F̂_g]
            δα_g = -P·A_g/F_g² · δF_g
            δβ_g = -P·A_g/F_g² · δe_g + 2·P·A_g·e_g/F_g³ · δF_g

        Parameters
        ----------
        dm1 : ndarray of shape (nao, nao) or (nset, nao, nao)
            First-order density matrix perturbation(s).

        Returns
        -------
        vmat : ndarray, same shape as dm1
            Linear response Fock matrix contribution(s).
        """
        if self.forces is None:
            raise RuntimeError(
                'kernel() must be called before _B_dot_x(). '
                'Forces have not been computed.')

        out_shape = dm1.shape
        nao = self.mol.nao_nr()
        dm1 = np.asarray(dm1).reshape(-1, nao, nao)
        nset = dm1.shape[0]

        if self.direct:
            vmat = self._B_dot_x_direct(dm1, nset)
        else:
            vmat = self._B_dot_x_cached(dm1, nset)

        return vmat.reshape(out_shape)

    def _B_dot_x_cached(self, dm1, nset):
        """Cached-mode implementation of _B_dot_x."""
        nao = self.mol.nao_nr()
        nao2 = nao * nao
        P = self.pressure_au

        Fg = self.forces
        eg = self.gtilde_expval
        Ag = self.areas

        gtilde_2d = self.gtilde.reshape(nao2, -1, order='F')
        fhat_2d = self.force_operators.reshape(nao2, -1, order='F')

        # Common factor: P * A_g / F_g^2
        pA_over_F2 = P * Ag / (Fg ** 2)
        # Factor for second term of δβ: 2 * P * A_g * e_g / F_g^3
        two_pAe_over_F3 = 2.0 * P * Ag * eg / (Fg ** 3)

        # Masked grid points: alpha is hard-clamped to 0, F frozen at 1.0.
        # Their derivatives must not contribute to the response.
        mask = (self.amplitudes == 0.0)
        if mask.any():
            pA_over_F2[mask] = 0.0
            two_pAe_over_F3[mask] = 0.0

        vmat = np.zeros((nset, nao, nao))
        for i in range(nset):
            dm1_flat = dm1[i].ravel(order='F')
            # Perturbed traces
            d_eg = dm1_flat @ gtilde_2d     # (ngrids,)
            d_Fg = dm1_flat @ fhat_2d       # (ngrids,)

            # Perturbed amplitudes
            d_alpha = -pA_over_F2 * d_Fg
            d_beta = -pA_over_F2 * d_eg + two_pAe_over_F3 * d_Fg

            # Assemble response Fock matrix
            vmat[i] = (gtilde_2d @ d_alpha + fhat_2d @ d_beta
                       ).reshape(nao, nao, order='F')

        return vmat

    def _B_dot_x_direct(self, dm1, nset):
        """Integral-direct mode implementation of _B_dot_x."""
        mol = self.mol
        nao = mol.nao_nr()
        nao2 = nao * nao
        P = self.pressure_au

        Fg = self.forces
        eg = self.gtilde_expval
        Ag = self.areas

        # Common factors
        pA_over_F2 = P * Ag / (Fg ** 2)
        two_pAe_over_F3 = 2.0 * P * Ag * eg / (Fg ** 3)

        # Masked grid points: alpha is hard-clamped to 0, F frozen at 1.0.
        # Their derivatives must not contribute to the response.
        mask = (self.amplitudes == 0.0)
        if mask.any():
            pA_over_F2[mask] = 0.0
            two_pAe_over_F3[mask] = 0.0

        # Chunking (same as _kernel_direct)
        max_memreq = 5 * self.n_gaussian * nao2 * 8.0 / 1e6
        max_memory = max(2000, mol.max_memory * 0.9 - lib.current_memory()[0])
        n_chunks = max(1, int(max_memreq // max_memory + 1))

        shells = np.arange(self.n_gaussian)
        chunks = np.array_split(shells, n_chunks)

        gmol = fakemol_for_gaussian(self.grid_coords, self.widths)
        gmol_p = fakemol_for_gaussian(
            self.grid_coords, self.widths, l=1, coeffs=2.0 * self.widths)
        supermol = mol + gmol
        supermol_p = mol + gmol_p

        vmat = np.zeros((nset, nao, nao))

        for shell_slice in chunks:
            off1 = int(shell_slice[0])
            off2 = len(shell_slice)
            slices = (0, mol.nbas, 0, mol.nbas,
                      mol.nbas + off1, mol.nbas + off1 + off2)

            overlap3_s = supermol.intor('int3c1e', shls_slice=slices, aosym='s1')
            overlap3_p = supermol_p.intor(
                'int3c1e', shls_slice=slices, aosym='s1'
            ).reshape(nao, nao, -1, 3)

            normals = self.surface_normals[shell_slice]
            force_ops = np.einsum('bkgc,gc->bkg', overlap3_p, normals,
                                  optimize=False)

            gtilde_2d = overlap3_s.reshape(nao2, -1, order='F')
            fhat_2d = force_ops.reshape(nao2, -1, order='F')

            pA_F2_c = pA_over_F2[shell_slice]
            two_pAe_F3_c = two_pAe_over_F3[shell_slice]

            for i in range(nset):
                dm1_flat = dm1[i].ravel(order='F')
                d_eg = dm1_flat @ gtilde_2d
                d_Fg = dm1_flat @ fhat_2d

                d_alpha = -pA_F2_c * d_Fg
                d_beta = -pA_F2_c * d_eg + two_pAe_F3_c * d_Fg

                vmat[i] += (gtilde_2d @ d_alpha + fhat_2d @ d_beta
                            ).reshape(nao, nao, order='F')

        return vmat

    def reset(self, mol=None):
        """Reset for geometry optimization / scanner."""
        if mol is not None:
            self.mol = mol
        self.__dict__.pop('gtilde', None)
        self.__dict__.pop('force_operators', None)
        self.e = None
        self.v = None
        self.amplitudes = None
        self.forces = None
        self.build()
        return self


def analytical_grad_vmat(gost, dm, atmlst=None):
    """Derivative of GOSTSHYP Fock matrix w.r.t. nuclear coordinates at fixed density.

    Computes dV/dR where V is the GOSTSHYP Fock matrix contribution,
    holding the density matrix D fixed. This provides the CP-HF right-hand
    side (make_h1) for the GOSTSHYP analytical Hessian.

    Parameters
    ----------
    gost : GOSTSHYP
        GOSTSHYP object with kernel() already called (needs amplitudes, forces, etc.)
    dm : ndarray of shape (nao, nao)
        Density matrix (held fixed during differentiation).
    atmlst : list of int, optional
        Atoms for which to compute derivatives. Default: all atoms.

    Returns
    -------
    dV : ndarray of shape (len(atmlst), 3, nao, nao)
        dV[ia, x] is the derivative of the Fock matrix w.r.t. coordinate x of atom atmlst[ia].
    """
    if gost.amplitudes is None:
        raise RuntimeError(
            'kernel() must be called before analytical_grad_vmat(). '
            'Amplitudes have not been computed.')

    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    from pyscf.solvent._gostshyp_hess import grad_vmat
    dV = grad_vmat(gost, dm)
    if atmlst is None:
        return dV
    return dV[np.asarray(list(atmlst), dtype=int)]


class WithGOSTSHYPHess:
    """Mixin that augments a vacuum Hessian with GOSTSHYP contributions."""

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithGOSTSHYPHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        # Enable equilibrium_solvation during CPHF so gen_response calls _B_dot_x
        logger.debug(self, 'Compute Hessian from solutes (with GOSTSHYP CPHF response)')
        with lib.temporary_env(self.base.with_solvent, equilibrium_solvation=True):
            self.de_solute = super().kernel(*args, **kwargs)

        logger.debug(self, 'Compute Hessian from solvents')
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        self.de_solvent = self.base.with_solvent.hess(dm)
        self.de = self.de_solute + self.de_solvent
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)

        solvent = self.base.with_solvent
        dm = self.base.make_rdm1(ao_repr=True)

        if isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            dm_tot = dm[0] + dm[1]
            dv = analytical_grad_vmat(solvent, dm_tot, atmlst=atmlst)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dv[i0]
                h1aob[i0] += dv[i0]
            return h1aoa, h1aob
        else:
            # RHF / RKS
            dv = analytical_grad_vmat(solvent, dm, atmlst=atmlst)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao

    def _finalize(self):
        pass


def make_hess_object(base_method):
    """Create a GOSTSHYP-augmented Hessian object from a GOSTSHYP-attached SCF.

    Parameters
    ----------
    base_method : SCF method
        A GOSTSHYP-attached SCF method (the result of mf.GOSTSHYP()).

    Returns
    -------
    hess : WithGOSTSHYPHess
        Hessian object with GOSTSHYP contributions.
    """
    from pyscf.solvent._attach_solvent import _Solvation
    from pyscf.hessian.rhf import HessianBase

    if isinstance(base_method, HessianBase):
        base_method = base_method.base

    if not isinstance(base_method, _Solvation):
        raise TypeError(
            f'make_hess_object requires a GOSTSHYP-attached SCF method; '
            f'got {base_method.__class__.__name__}')
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not available for energy hessian')

    vac_mf = base_method.undo_solvent()
    vac_hess = vac_mf.Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithGOSTSHYPHess(vac_hess),
                         (WithGOSTSHYPHess, vac_hess.__class__), name)


class SCFWithGOSTSHYP(_attach_solvent.SCFWithSolvent):
    """SCFWithSolvent that uses the GOSTSHYP Hessian instead of the PCM one.

    The override lives on the class (not the instance), so that
    undo_solvent() drops it together with SCFWithSolvent.
    """

    def Hessian(self):
        return make_hess_object(self)


def gostshyp_for_scf(mf, solvent_obj=None, dm=None):
    """Attach GOSTSHYP solvent model to SCF method."""
    if not isinstance(mf, scf.hf.SCF):
        raise TypeError(
            'GOSTSHYP can only be used with SCF methods (RHF/UHF/RKS/UKS). '
            f'Got {mf.__class__.__name__}')
    if solvent_obj is None:
        solvent_obj = GOSTSHYP(mf.mol)
    sol_mf = _attach_solvent._for_scf(mf, solvent_obj, dm)

    # Replace the inherited PCM Hessian hook with GOSTSHYP's own
    cls = sol_mf.__class__
    if not issubclass(cls, SCFWithGOSTSHYP):
        lib.set_class(sol_mf, (SCFWithGOSTSHYP,) + cls.__bases__[1:], cls.__name__)
    return sol_mf


# Inject GOSTSHYP into SCF classes
scf.hf.SCF.GOSTSHYP = gostshyp_for_scf
