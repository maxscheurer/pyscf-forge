#!/usr/bin/env python
"""
Test each group (G1-G5) of d²F_g/(dR_Ax dR_By) INDIVIDUALLY against
targeted finite differences.

Key insight: for VDW cavity, grid_coords[g] = atom_coords[atom_idx[g]] +
radius*scaling*normals[g]. When atom B moves, ALL its grid points co-move.
The G1 (pos×pos) test must reflect this.

Strategy:
  G1: Move atom B (AO centers + grid centers on B), keep ω fixed.
      fdiff of dF_pos gives the full position×position second derivative.
  G2+G3: Cross terms between position and width changes.
      Move atom B → ω changes via area change → captures cross.
      Verified by: fdiff of full dF_trace minus G1 minus G4 minus G5.
  G4: d²ω · dFhat_domega_trace. Perturb widths directly by small δ.
  G5: (dω)² · d²Fhat/dω². Perturb widths, measure second response of F.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import make_h2_system, make_h2o_system, get_gost_options
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian


# =============================================================================
# Helper: compute F_g at given geometry/grid/widths
# =============================================================================
def compute_Fg(mol_obj, grid_coords, widths, normals, dm):
    """F_g = Tr[D · Fhat_g] for p-type Gaussian at given grid/widths."""
    nao = mol_obj.nao_nr()
    ngrids = len(widths)
    gmol = fakemol_for_gaussian(grid_coords, widths, l=1, coeffs=2.0 * widths)
    supermol = mol_obj + gmol
    sl = (0, mol_obj.nbas, 0, mol_obj.nbas,
          mol_obj.nbas, mol_obj.nbas + gmol.nbas)
    raw = supermol.intor('int3c1e', shls_slice=sl).reshape(nao, nao, ngrids, 3)
    fhat = np.einsum('ijgc,gc->ijg', raw, normals, optimize=True)
    return np.einsum('ijg,ij->g', fhat, dm, optimize=True)


# =============================================================================
# Helper: compute dF_pos at given geometry/grid/widths
# =============================================================================
def compute_dF_pos(mol_obj, grid_coords, widths, normals, dm, atom_idx,
                   aoslice):
    """dF_g/dR_Ax from ip1 integrals. Includes bra/ket AND aux contributions."""
    nao = mol_obj.nao_nr()
    natm = mol_obj.natm
    ngrids = len(widths)

    gmol = fakemol_for_gaussian(grid_coords, widths, l=1, coeffs=2.0 * widths)
    supermol = mol_obj + gmol
    sl = (0, mol_obj.nbas, 0, mol_obj.nbas,
          mol_obj.nbas, mol_obj.nbas + gmol.nbas)
    sl_g = (mol_obj.nbas, mol_obj.nbas + gmol.nbas,
            0, mol_obj.nbas, 0, mol_obj.nbas)

    # Bra/ket ip1
    ip1_raw = supermol.intor('int3c1e_ip1', shls_slice=sl
                             ).reshape(3, nao, nao, ngrids, 3)
    ip1 = np.einsum('xijgc,gc->xijg', ip1_raw, normals, optimize=True)
    ip1_dm = (np.einsum('xijg,ij->xig', ip1, dm, optimize=True)
              + np.einsum('xijg,ji->xig', ip1, dm, optimize=True))

    # Aux ip1
    ip1_aux_raw = supermol.intor('int3c1e_ip1', shls_slice=sl_g
                                 ).reshape(3, ngrids, 3, nao, nao)
    ip1_aux = np.einsum('xgcij,gc->xgij', ip1_aux_raw, normals, optimize=True)
    ip1_aux_dm = np.einsum('xgij,ij->xg', ip1_aux, dm, optimize=True)

    dF = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dF[A] -= np.sum(ip1_dm[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        dF[atom_idx[g], :, g] -= ip1_aux_dm[:, g]
    return dF


# =============================================================================
# TEST G1: Position × Position (grid centers co-move with atoms)
# =============================================================================
def test_G1(gost, dm, mol, label=""):
    """G1 test: displace atom B, co-move its grid points, keep ω fixed.
    fdiff of dF_pos gives the position×position second derivative."""
    print(f"\n{'='*60}")
    print(f"TEST G1: Position × Position [{label}]")
    print(f"{'='*60}")

    natm = mol.natm
    ngrids = gost.n_gaussian
    widths = gost.widths.copy()
    normals = gost.surface_normals.copy()
    aoslice = mol.aoslice_by_atom()
    atom_idx = gost.atom_idx.copy()
    coords0 = mol.atom_coords().copy()
    gc0 = gost.grid_coords.copy()

    step = 1e-5
    d2F_G1_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            # + step: move atom B and its grid points
            coords_p = coords0.copy()
            coords_p[B, y] += step
            gc_p = gc0.copy()
            gc_p[atom_idx == B, y] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            dF_p = compute_dF_pos(mol_p, gc_p, widths, normals, dm,
                                  atom_idx, aoslice)

            # - step
            coords_m = coords0.copy()
            coords_m[B, y] -= step
            gc_m = gc0.copy()
            gc_m[atom_idx == B, y] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            dF_m = compute_dF_pos(mol_m, gc_m, widths, normals, dm,
                                  atom_idx, aoslice)

            d2F_G1_fd[:, B, :, y, :] = (dF_p - dF_m) / (2 * step)

    # Analytical G1 from 04_d2F
    from importlib import import_module
    m = import_module('04_d2F')
    # Get full analytical (with guard) and G1-only (thresh=1.0 masks all ω terms)
    d2F_full = m.compute_d2F_analytical(gost, dm, mol, force_thresh=0)
    d2F_g1_only = m.compute_d2F_analytical(gost, dm, mol, force_thresh=1.0)

    err = np.max(np.abs(d2F_g1_only - d2F_G1_fd))
    ref = np.max(np.abs(d2F_G1_fd))
    print(f"  G1 fdiff ref:  {ref:.3e}")
    print(f"  G1 abs error:  {err:.3e}")

    if err < 1e-6:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        # Show per-grid-point breakdown
        per_g = np.max(np.abs(d2F_g1_only - d2F_G1_fd), axis=(0, 1, 2, 3))
        worst = np.argsort(per_g)[-5:][::-1]
        for g in worst:
            if per_g[g] > 1e-10:
                print(f"    g={g}: err={per_g[g]:.3e}, F={gost.forces[g]:.3e}")

    return d2F_g1_only, d2F_G1_fd


# =============================================================================
# TEST G4+G5: Pure width response (perturb ω directly)
# =============================================================================
def test_G4_G5(gost, dm, mol, label=""):
    """G4+G5 test: perturb widths directly (δω), measure d²F/dω².
    G4: second-order area contribution
    G5: (dω)² · d²F/dω²

    We fdiff F_g(ω + δ) to get dF/dω, then fdiff again to get d²F/dω².
    Compare against analytical dFhat_domega_trace and d2Fhat_domega2_trace."""
    print(f"\n{'='*60}")
    print(f"TEST G4+G5: Width response (direct ω perturbation) [{label}]")
    print(f"{'='*60}")

    ngrids = gost.n_gaussian
    widths = gost.widths.copy()
    normals = gost.surface_normals.copy()
    gc = gost.grid_coords.copy()

    # --- First derivative: dF/dω by fdiff ---
    step = 1e-4  # relative to width
    dF_domega_fd = np.zeros(ngrids)
    d2F_domega2_fd = np.zeros(ngrids)

    for g in range(ngrids):
        w0 = widths[g]
        h = max(step * w0, 1e-6)  # absolute step

        wp = widths.copy(); wp[g] = w0 + h
        wm = widths.copy(); wm[g] = w0 - h
        w2p = widths.copy(); w2p[g] = w0 + 2*h
        w2m = widths.copy(); w2m[g] = w0 - 2*h

        Fp = compute_Fg(mol, gc, wp, normals, dm)[g]
        Fm = compute_Fg(mol, gc, wm, normals, dm)[g]
        F0 = compute_Fg(mol, gc, widths, normals, dm)[g]

        dF_domega_fd[g] = (Fp - Fm) / (2 * h)
        d2F_domega2_fd[g] = (Fp - 2*F0 + Fm) / (h**2)

    # Analytical values
    from importlib import import_module
    m = import_module('04_d2F')
    f_dm = m.compute_f_contracted_dm(gost, dm, mol)
    h5_dm = m.compute_h5_contracted_dm(gost, dm, mol)
    forces = gost.forces

    dF_domega_ana = forces / widths + f_dm
    d2F_domega2_ana = 2.0 * f_dm / widths + h5_dm

    # Compare (only at stable points)
    stable = forces > 1e-9
    err1 = np.max(np.abs((dF_domega_ana - dF_domega_fd)[stable]))
    ref1 = np.max(np.abs(dF_domega_fd[stable]))
    err2 = np.max(np.abs((d2F_domega2_ana - d2F_domega2_fd)[stable]))
    ref2 = np.max(np.abs(d2F_domega2_fd[stable]))

    print(f"  dF/dω:  err={err1:.3e}, ref={ref1:.3e} ({stable.sum()} stable pts)")
    print(f"  d²F/dω²: err={err2:.3e}, ref={ref2:.3e}")

    if err1 < 1e-6 and err2 < 1e-6:
        print("  PASS ✓")
    elif err1 < 1e-6:
        print("  dF/dω PASS ✓, d²F/dω² needs checking")
        # Show worst points for d²F/dω²
        per_g = np.abs(d2F_domega2_ana - d2F_domega2_fd)
        per_g[~stable] = 0
        worst = np.argsort(per_g)[-5:][::-1]
        for g in worst:
            if per_g[g] > 1e-8:
                print(f"    g={g}: err={per_g[g]:.3e}, ana={d2F_domega2_ana[g]:.3e}, "
                      f"fd={d2F_domega2_fd[g]:.3e}, w={widths[g]:.1f}")
    else:
        print("  FAIL ✗")
        per_g = np.abs(dF_domega_ana - dF_domega_fd)
        per_g[~stable] = 0
        worst = np.argsort(per_g)[-5:][::-1]
        for g in worst:
            if per_g[g] > 1e-8:
                print(f"    g={g}: err={per_g[g]:.3e}, F={forces[g]:.3e}, w={widths[g]:.1f}")

    return dF_domega_ana, dF_domega_fd, d2F_domega2_ana, d2F_domega2_fd


# =============================================================================
# TEST G2+G3: Cross terms (position × width)
# =============================================================================
def test_G2_G3(gost, dm, mol, label=""):
    """G2+G3: cross between position and width changes.

    Strategy: compute the FULL d²F by fdiff (moving everything together),
    subtract G1 (position-only fdiff) and G4+G5 (width-only, from analytical
    since width changes are through area → ω coupling).
    The remainder = G2+G3.

    Alternatively: fdiff of dF_pos w.r.t. ω changes.
    d_dFhat_domega_dm_pos = (1/ω)·dF_pos + d_f_dm_pos

    Test: perturb ω at one grid point, see how dF_pos changes.
    """
    print(f"\n{'='*60}")
    print(f"TEST G2+G3: Position × Width cross terms [{label}]")
    print(f"{'='*60}")

    natm = mol.natm
    ngrids = gost.n_gaussian
    widths = gost.widths.copy()
    normals = gost.surface_normals.copy()
    gc = gost.grid_coords.copy()
    aoslice = mol.aoslice_by_atom()
    atom_idx = gost.atom_idx.copy()

    # d_dFhat_domega_dm_pos[A,x,g] = d(dF_pos[A,x,g])/dω_g
    # fdiff: perturb ω_g, recompute dF_pos, measure change
    step_frac = 1e-4
    d_dF_domega_fd = np.zeros((natm, 3, ngrids))

    for g in range(ngrids):
        w0 = widths[g]
        h = max(step_frac * w0, 1e-6)

        wp = widths.copy(); wp[g] = w0 + h
        wm = widths.copy(); wm[g] = w0 - h

        dF_p = compute_dF_pos(mol, gc, wp, normals, dm, atom_idx, aoslice)
        dF_m = compute_dF_pos(mol, gc, wm, normals, dm, atom_idx, aoslice)

        d_dF_domega_fd[:, :, g] = (dF_p[:, :, g] - dF_m[:, :, g]) / (2 * h)

    # Analytical: d_dFhat_domega_dm_pos = dF_pos/ω + d_f_dm_pos
    # Get this from 04_d2F internals
    from importlib import import_module
    m = import_module('04_d2F')

    # Compute dF_pos at reference
    dF_pos_ref = compute_dF_pos(mol, gc, widths, normals, dm, atom_idx, aoslice)

    # Compute d_f_dm_pos (ip1 of f-type l=3)
    # We need to extract this from 04_d2F. Let me compute it directly.
    nao_cart = mol.nao_nr(cart=True)
    c2s = mol.cart2sph_coeff(normalized='sp') if not mol.cart else None
    nao = mol.nao_nr()

    gmol_f = fakemol_for_gaussian(gc, widths, l=3, coeffs=-2.0 * widths)
    supermol_f = mol + gmol_f
    supermol_f.cart = True
    sl_f = (0, mol.nbas, 0, mol.nbas,
            mol.nbas, mol.nbas + gmol_f.nbas)
    sl_fg = (mol.nbas, mol.nbas + gmol_f.nbas,
             0, mol.nbas, 0, mol.nbas)

    # Bra ip1 of f-type
    ip1_f_bra = supermol_f.intor('int3c1e_ip1', shls_slice=sl_f
                                 ).reshape(3, nao_cart, nao_cart, ngrids, 10)
    ip1_fx = ip1_f_bra[:,:,:,:,0] + ip1_f_bra[:,:,:,:,3] + ip1_f_bra[:,:,:,:,5]
    ip1_fy = ip1_f_bra[:,:,:,:,1] + ip1_f_bra[:,:,:,:,6] + ip1_f_bra[:,:,:,:,8]
    ip1_fz = ip1_f_bra[:,:,:,:,2] + ip1_f_bra[:,:,:,:,7] + ip1_f_bra[:,:,:,:,9]
    del ip1_f_bra
    ip1_f_bra_contracted = (ip1_fx * normals[:, 0] + ip1_fy * normals[:, 1]
                            + ip1_fz * normals[:, 2])
    del ip1_fx, ip1_fy, ip1_fz
    if c2s is not None:
        ip1_f_bra_contracted = np.einsum('mi,xijg,jn->xmng', c2s.T,
                                         ip1_f_bra_contracted, c2s,
                                         optimize=True)
    ip1_f_dm_bra = (np.einsum('xijg,ij->xig', ip1_f_bra_contracted, dm,
                              optimize=True)
                    + np.einsum('xijg,ji->xig', ip1_f_bra_contracted, dm,
                                optimize=True))
    del ip1_f_bra_contracted

    # Aux ip1 of f-type
    ip1_f_aux = supermol_f.intor('int3c1e_ip1', shls_slice=sl_fg
                                 ).reshape(3, ngrids, 10, nao_cart, nao_cart)
    ip1_fx_a = ip1_f_aux[:,:,0,:,:] + ip1_f_aux[:,:,3,:,:] + ip1_f_aux[:,:,5,:,:]
    ip1_fy_a = ip1_f_aux[:,:,1,:,:] + ip1_f_aux[:,:,6,:,:] + ip1_f_aux[:,:,8,:,:]
    ip1_fz_a = ip1_f_aux[:,:,2,:,:] + ip1_f_aux[:,:,7,:,:] + ip1_f_aux[:,:,9,:,:]
    del ip1_f_aux
    ip1_f_aux_contracted = (ip1_fx_a * normals[:, 0][:, None, None]
                            + ip1_fy_a * normals[:, 1][:, None, None]
                            + ip1_fz_a * normals[:, 2][:, None, None])
    del ip1_fx_a, ip1_fy_a, ip1_fz_a
    if c2s is not None:
        ip1_f_aux_contracted = np.einsum('mi,xgij,jn->xgmn', c2s.T,
                                         ip1_f_aux_contracted, c2s,
                                         optimize=True)
    ip1_f_dm_aux = np.einsum('xgij,ij->xg', ip1_f_aux_contracted, dm,
                             optimize=True)
    del ip1_f_aux_contracted

    d_f_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_f_dm_pos[A] -= np.sum(ip1_f_dm_bra[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        d_f_dm_pos[atom_idx[g], :, g] -= ip1_f_dm_aux[:, g]

    # Analytical cross-derivative
    d_dF_domega_ana = dF_pos_ref / widths + d_f_dm_pos

    # Compare
    forces = gost.forces
    stable = forces > 1e-9
    err = np.max(np.abs((d_dF_domega_ana - d_dF_domega_fd)[:, :, stable]))
    ref = np.max(np.abs(d_dF_domega_fd[:, :, stable]))
    print(f"  d(dF_pos)/dω: err={err:.3e}, ref={ref:.3e}")

    if err < 1e-6:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        per_g = np.max(np.abs(d_dF_domega_ana - d_dF_domega_fd), axis=(0, 1))
        per_g[~stable] = 0
        worst = np.argsort(per_g)[-5:][::-1]
        for g in worst:
            if per_g[g] > 1e-8:
                print(f"    g={g}: err={per_g[g]:.3e}, F={forces[g]:.3e}, w={widths[g]:.1f}")

    return d_dF_domega_ana, d_dF_domega_fd


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("H2/sto-3g")
    print("=" * 60)
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points")
    test_G1(gost, dm, mol, "H2")
    test_G4_G5(gost, dm, mol, "H2")
    test_G2_G3(gost, dm, mol, "H2")

    print("\n\n" + "=" * 60)
    print("H2O/cc-pVDZ")
    print("=" * 60)
    gost, dm, mol = make_h2o_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points")
    test_G1(gost, dm, mol, "H2O")
    test_G4_G5(gost, dm, mol, "H2O")
    test_G2_G3(gost, dm, mol, "H2O")
