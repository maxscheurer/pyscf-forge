"""
Common setup for Hessian term-by-term testing.

Provides a small H2/sto-3g system with converged density for fast iteration.
All scripts import from here to ensure consistency.
"""

import numpy as np
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA


def make_h2_system(cavity='vdw', npoints=110, scaling_factor=1.2,
                   pressure_mpa=50_000):
    """Set up H2/sto-3g with GOSTSHYP, return (gost, dm, mol)."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='sto-3g',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    options = {
        'cavity': cavity,
        'pressure_mpa': pressure_mpa,
        'npoints': npoints,
        'scaling_factor': scaling_factor,
    }
    gost = GOSTSHYP(mol, options=options)
    gost.kernel(dm)

    return gost, dm, mol


def make_h2o_system(cavity='vdw', npoints=110, scaling_factor=1.2,
                    pressure_mpa=50_000):
    """Set up H2O/cc-pVDZ with GOSTSHYP, return (gost, dm, mol)."""
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    options = {
        'cavity': cavity,
        'pressure_mpa': pressure_mpa,
        'npoints': npoints,
        'scaling_factor': scaling_factor,
    }
    gost = GOSTSHYP(mol, options=options)
    gost.kernel(dm)

    return gost, dm, mol


def compute_scalar_traces(gost, dm):
    """Compute de_g/dR and dF_g/dR for all atoms.

    Returns
    -------
    dg_trace : ndarray (natm, 3, ngrids)
        de_g/dR_Ax — first derivative of Gtilde trace.
    dF_trace : ndarray (natm, 3, ngrids)
        dF_g/dR_Ax — first derivative of Fhat trace.
    """
    mol = gost.mol
    nao = mol.nao_nr()
    nao_cart = mol.nao_nr(cart=True)
    natm = mol.natm
    ngrids = gost.n_gaussian
    aoslice = mol.aoslice_by_atom()

    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized='sp')

    # Get area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas.transpose(1, 2, 0) * gost._occ_ratio_sq
    else:
        _, dareas = get_dF_dA(gost.surface_dict)
        dareas = dareas.transpose(1, 2, 0)  # (natm, 3, ngrids)

    widths = gost.widths
    areas = gost.areas
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)  # dω/dA

    # --- de_g/dR (Gtilde trace derivative) ---
    gmol_s = fakemol_for_gaussian(gost.grid_coords, widths)
    supermol_s = mol + gmol_s
    slices_s = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_s.nbas)
    slices_sg = (mol.nbas, mol.nbas + gmol_s.nbas, 0, mol.nbas, 0, mol.nbas)

    # ip1 bra/ket
    dPQ_s = supermol_s.intor('int3c1e_ip1', shls_slice=slices_s)
    # ip1 Gaussian center
    dG_s = supermol_s.intor('int3c1e_ip1', shls_slice=slices_sg)

    dg_trace = np.zeros((natm, 3, ngrids))

    # Bra/ket trace
    dPQ_s_dm = np.einsum('xijg,ij->xig', dPQ_s, dm, optimize=True)
    dPQ_s_dm += np.einsum('xijg,ji->xig', dPQ_s, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dg_trace[A, :, :] -= np.sum(dPQ_s_dm[:, p0:p1, :], axis=1)

    # Gaussian center trace
    dG_s_dm = np.einsum('xgij,ij->xg', dG_s, dm, optimize=True)
    for g in range(ngrids):
        atom_g = gost.atom_idx[g]
        dg_trace[atom_g, :, g] -= dG_s_dm[:, g]

    # Width trace: wgrad * dA * Tr[D * ∂Gtilde/∂ω]
    # ∂Gtilde/∂ω = -diagd (from d-type integrals)
    gmol_d = fakemol_for_gaussian(gost.grid_coords, widths, l=2)
    supermol_d = mol + gmol_d
    supermol_d.cart = True
    slices_d = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_d.nbas)
    overlap3d = supermol_d.intor('int3c1e', shls_slice=slices_d
                                 ).reshape(nao_cart, nao_cart, ngrids, 6)
    if not mol.cart:
        overlap3d = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s,
                              optimize=True)
    diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
    diagd_dm = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
    # de/dω = -diagd_dm (trace of D with ∂Gtilde/∂ω)
    # de/dR via width = dω/dA * dA/dR * de/dω = wgrad * dA * (-diagd_dm)
    dg_trace -= np.einsum('axg,g->axg', dareas, wgrad_prefs * diagd_dm,
                          optimize=True)

    # --- dF_g/dR (Fhat trace derivative) ---
    gmol_p = fakemol_for_gaussian(gost.grid_coords, widths, l=1,
                                  coeffs=2.0 * widths)
    supermol_p = mol + gmol_p
    slices_p = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)
    slices_pg = (mol.nbas, mol.nbas + gmol_p.nbas, 0, mol.nbas, 0, mol.nbas)

    # p-type ip1 bra/ket contracted with normals
    dPQ_p_raw = supermol_p.intor('int3c1e_ip1', shls_slice=slices_p
                                 ).reshape(3, nao, nao, ngrids, 3)
    dPQ_p = np.einsum('xijgc,gc->xijg', dPQ_p_raw, gost.surface_normals,
                      optimize=True)

    # p-type ip1 Gaussian center contracted with normals
    dG_p_raw = supermol_p.intor('int3c1e_ip1', shls_slice=slices_pg
                                ).reshape(3, ngrids, 3, nao, nao)
    dG_p = np.einsum('xgcij,gc->xgij', dG_p_raw, gost.surface_normals,
                     optimize=True)

    dF_trace = np.zeros((natm, 3, ngrids))

    # Bra/ket trace
    dPQ_p_dm = np.einsum('xijg,ij->xig', dPQ_p, dm, optimize=True)
    dPQ_p_dm += np.einsum('xijg,ji->xig', dPQ_p, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dF_trace[A, :, :] -= np.sum(dPQ_p_dm[:, p0:p1, :], axis=1)

    # Gaussian center trace
    dG_p_dm = np.einsum('xgij,ij->xg', dG_p, dm, optimize=True)
    for g in range(ngrids):
        atom_g = gost.atom_idx[g]
        dF_trace[atom_g, :, g] -= dG_p_dm[:, g]

    # Width trace for Fhat: dω/dA * dA/dR * Tr[D * ∂Fhat/∂ω]
    # ∂Fhat/∂ω = Fhat/ω + f_contracted (from f-type integrals)
    forces = gost.forces
    gmol_ft = fakemol_for_gaussian(gost.grid_coords, widths, l=3,
                                   coeffs=-2.0 * widths)
    supermol_ft = mol + gmol_ft
    supermol_ft.cart = True
    slices_ft = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_ft.nbas)
    overlap3f = supermol_ft.intor('int3c1e', shls_slice=slices_ft
                                  ).reshape(nao_cart, nao_cart, ngrids, 10)
    if not mol.cart:
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s,
                              optimize=True)
    fx = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    fy = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    fz = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
    f_contracted_dm = (
        np.einsum('ijg,ij->g', fx, dm, optimize=True) * gost.surface_normals[:, 0]
        + np.einsum('ijg,ij->g', fy, dm, optimize=True) * gost.surface_normals[:, 1]
        + np.einsum('ijg,ij->g', fz, dm, optimize=True) * gost.surface_normals[:, 2]
    )
    # Tr[D * ∂Fhat/∂ω] = F/ω + f_contracted_dm
    dFhat_domega_trace = forces / widths + f_contracted_dm
    # dF/dR via width = wgrad * dA * (F/ω + f_contracted_dm)
    dF_trace += np.einsum('axg,g->axg', dareas,
                          wgrad_prefs * dFhat_domega_trace, optimize=True)

    return dg_trace, dF_trace


def compute_eg_Fg_at_geom(mol, dm, gost_opts, coords):
    """Compute e_g and F_g at a given geometry (for fdiff).

    Parameters
    ----------
    mol : template molecule
    dm : density matrix (fixed)
    gost_opts : dict of GOSTSHYP options
    coords : ndarray (natm, 3) — new coordinates in Bohr

    Returns
    -------
    e_g : ndarray (ngrids,) — Gtilde trace
    F_g : ndarray (ngrids,) — Fhat trace (forces)
    """
    mol_new = mol.copy()
    mol_new.set_geom_(coords, unit='Bohr')
    gost_new = GOSTSHYP(mol_new, options=gost_opts)
    gost_new.kernel(dm)
    return gost_new.gtilde_expval.copy(), gost_new.forces.copy()


def get_gost_options(gost):
    """Extract options dict from a GOSTSHYP object."""
    opts = {
        'cavity': gost.cavity,
        'pressure_mpa': gost.pressure_mpa,
        'npoints': gost.npoints,
        'scaling_factor': gost.scaling_factor,
    }
    if gost.cavity == 'vdw/occ':
        opts['r_ext'] = gost.r_ext
    return opts
