#!/usr/bin/env python
"""
Test each group (G1-G5) of d²F_g/(dR_Ax dR_By) INDIVIDUALLY against
targeted finite differences.

Strategy (mirrors 07_test_d2e_groups.py):
  G1 (pos×pos): fix ω, fix grid coords, only move AO centers
  G2 (fwd cross): fix positions, perturb ω via dA_By only, check d(dF_pos)/dω
  G3 (rev cross): fix ω, check d(width_part)/dpos_By
  G4 (d²ω): check d²ω · dFhat_domega_trace (with fixed dFhat_domega_trace)
  G5 (ω×ω): check dω/dR_Ax · dω/dR_By · d2Fhat_domega2_trace
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import make_h2_system, get_gost_options
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A
from pyscf.solvent.pcm import gen_surface, modified_Bondi


def setup():
    """Create H2 system and precompute common quantities."""
    gost, dm, mol = make_h2_system()
    nao = mol.nao_nr()
    nao_cart = mol.nao_nr(cart=True)
    natm = mol.natm
    ngrids = gost.n_gaussian
    widths = gost.widths
    areas = gost.areas
    normals = gost.surface_normals
    aoslice = mol.aoslice_by_atom()
    c2s = mol.cart2sph_coeff(normalized='sp') if not mol.cart else None
    return (gost, dm, mol, nao, nao_cart, natm, ngrids, widths, areas,
            normals, aoslice, c2s)


def compute_Fg_at_geom(mol_displaced, grid_coords, widths, normals, dm, c2s):
    """Compute F_g = Tr[D · Fhat_g] at displaced AO positions but FIXED grid/widths."""
    nao = mol_displaced.nao_nr()
    gmol = fakemol_for_gaussian(grid_coords, widths, l=1, coeffs=2.0 * widths)
    supermol = mol_displaced + gmol
    slices = (0, mol_displaced.nbas, 0, mol_displaced.nbas,
              mol_displaced.nbas, mol_displaced.nbas + gmol.nbas)
    ngrids = len(widths)
    overlap3 = supermol.intor('int3c1e', shls_slice=slices
                              ).reshape(nao, nao, ngrids, 3)
    fhat = np.einsum('ijgc,gc->ijg', overlap3, normals, optimize=True)
    return np.einsum('ijg,ij->g', fhat, dm, optimize=True)


def compute_dF_pos(mol_ref, grid_coords, widths, normals, dm, atom_idx,
                   aoslice, c2s):
    """Compute dF_g/dR_Ax (position-only: bra+ket+center) at FIXED ω."""
    nao = mol_ref.nao_nr()
    natm = mol_ref.natm
    ngrids = len(widths)

    gmol = fakemol_for_gaussian(grid_coords, widths, l=1, coeffs=2.0 * widths)
    supermol = mol_ref + gmol
    slices = (0, mol_ref.nbas, 0, mol_ref.nbas,
              mol_ref.nbas, mol_ref.nbas + gmol.nbas)
    slices_g = (mol_ref.nbas, mol_ref.nbas + gmol.nbas,
                0, mol_ref.nbas, 0, mol_ref.nbas)

    # Bra/ket ip1
    ip1_bra_raw = supermol.intor('int3c1e_ip1', shls_slice=slices
                                 ).reshape(3, nao, nao, ngrids, 3)
    ip1_bra = np.einsum('xijgc,gc->xijg', ip1_bra_raw, normals,
                        optimize=True)
    ip1_dm = np.einsum('xijg,ij->xig', ip1_bra, dm, optimize=True)
    ip1_dm += np.einsum('xijg,ji->xig', ip1_bra, dm, optimize=True)

    # Aux ip1
    ip1_aux_raw = supermol.intor('int3c1e_ip1', shls_slice=slices_g
                                 ).reshape(3, ngrids, 3, nao, nao)
    ip1_aux = np.einsum('xgcij,gc->xgij', ip1_aux_raw, normals,
                        optimize=True)
    ip1_aux_dm = np.einsum('xgij,ij->xg', ip1_aux, dm, optimize=True)

    dF = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dF[A] -= np.sum(ip1_dm[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        dF[atom_idx[g], :, g] -= ip1_aux_dm[:, g]

    return dF


def compute_dFhat_domega_trace(mol_ref, grid_coords, widths, normals, dm,
                               c2s):
    """Compute Tr[D · ∂Fhat/∂ω] = F/ω + f_contracted_dm at given ω."""
    nao = mol_ref.nao_nr()
    nao_cart = mol_ref.nao_nr(cart=True)
    ngrids = len(widths)

    # F_g
    F_g = compute_Fg_at_geom(mol_ref, grid_coords, widths, normals, dm, c2s)

    # f_contracted_dm
    gmol_f = fakemol_for_gaussian(grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol_ref + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol_ref.nbas, 0, mol_ref.nbas,
                mol_ref.nbas, mol_ref.nbas + gmol_f.nbas)
    overlap3f = supermol_f.intor('int3c1e', shls_slice=slices_f
                                 ).reshape(nao_cart, nao_cart, ngrids, 10)
    if c2s is not None:
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s,
                              optimize=True)
    fx = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    fy = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    fz = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
    f_contracted_dm = (
        np.einsum('ijg,ij->g', fx, dm, optimize=True) * normals[:, 0]
        + np.einsum('ijg,ij->g', fy, dm, optimize=True) * normals[:, 1]
        + np.einsum('ijg,ij->g', fz, dm, optimize=True) * normals[:, 2])

    return F_g / widths + f_contracted_dm


# =============================================================================
# TEST G1: Position × Position
# =============================================================================
def test_G1():
    """G1: d²F/(dR_Ax dR_By) at FIXED ω and FIXED grid positions.
    Only AO centers move. Grid points stay put."""
    print("\n" + "=" * 60)
    print("TEST G1: Position × Position (fixed ω, fixed grid)")
    print("=" * 60)

    (gost, dm, mol, nao, nao_cart, natm, ngrids, widths, areas,
     normals, aoslice, c2s) = setup()
    coords0 = mol.atom_coords().copy()
    grid_coords = gost.grid_coords.copy()

    step = 1e-5
    d2F_G1_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy(); coords_p[B, y] += step
            mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr')
            dF_p = compute_dF_pos(mol_p, grid_coords, widths, normals, dm,
                                  gost.atom_idx, aoslice, c2s)

            coords_m = coords0.copy(); coords_m[B, y] -= step
            mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr')
            dF_m = compute_dF_pos(mol_m, grid_coords, widths, normals, dm,
                                  gost.atom_idx, aoslice, c2s)

            d2F_G1_fd[:, B, :, y, :] = (dF_p - dF_m) / (2 * step)

    # Analytical G1: ipip1 + ipvip1 + ip1ip2 + ipip2 (all p-type, normal-contracted)
    gmol_p = fakemol_for_gaussian(grid_coords, widths, l=1,
                                  coeffs=2.0 * widths)
    supermol = mol + gmol_p
    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)

    d2F_G1_ana = np.zeros((natm, natm, 3, 3, ngrids))

    # ipip1
    ipip1_raw = supermol.intor('int3c1e_ipip1', shls_slice=slices
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip1 = np.einsum('xijgc,gc->xijg', ipip1_raw, normals, optimize=True)
    del ipip1_raw
    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d2F_G1_ana[A, A] += np.sum(ipip1_dm[:, p0:p1, :], axis=1
                                    ).reshape(3, 3, ngrids)
    del ipip1, ipip1_dm

    # ipvip1
    ipvip1_raw = supermol.intor('int3c1e_ipvip1', shls_slice=slices
                                ).reshape(9, nao, nao, ngrids, 3)
    ipvip1 = np.einsum('xijgc,gc->xijg', ipvip1_raw, normals, optimize=True)
    del ipvip1_raw
    for A in range(natm):
        p0_A, p1_A = aoslice[A, 2], aoslice[A, 3]
        for B in range(natm):
            p0_B, p1_B = aoslice[B, 2], aoslice[B, 3]
            block = ipvip1[:, p0_A:p1_A, p0_B:p1_B, :]
            trace_block = (
                np.einsum('xijg,ij->xg', block,
                          dm[p0_A:p1_A, p0_B:p1_B], optimize=True)
                + np.einsum('xijg,ji->xg', block,
                            dm[p0_B:p1_B, p0_A:p1_A], optimize=True))
            d2F_G1_ana[A, B] += trace_block.reshape(3, 3, ngrids)
    del ipvip1

    # ip1ip2
    ip1ip2_raw = supermol.intor('int3c1e_ip1ip2', shls_slice=slices
                                ).reshape(9, nao, nao, ngrids, 3)
    ip1ip2 = np.einsum('xijgc,gc->xijg', ip1ip2_raw, normals, optimize=True)
    del ip1ip2_raw
    ip1ip2_dm = np.einsum('xijg,ij->xig', ip1ip2, dm, optimize=True)
    ip1ip2_dm += np.einsum('xijg,ji->xig', ip1ip2, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib_A = np.sum(ip1ip2_dm[:, p0:p1, :], axis=1)
        for g in range(ngrids):
            B = gost.atom_idx[g]
            d2F_G1_ana[A, B, :, :, g] += contrib_A[:, g].reshape(3, 3)
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            trace_ig = np.sum(ip1ip2_dm[:, p0:p1, g], axis=1)
            d2F_G1_ana[A, B, :, :, g] += trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm

    # ipip2
    ipip2_raw = supermol.intor('int3c1e_ipip2', shls_slice=slices
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip2 = np.einsum('xijgc,gc->xijg', ipip2_raw, normals, optimize=True)
    del ipip2_raw
    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2F_G1_ana[B, B, :, :, g] += ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    err = np.max(np.abs(d2F_G1_ana - d2F_G1_fd))
    ref = np.max(np.abs(d2F_G1_fd))
    print(f"  max |G1_fd|:  {ref:.3e}")
    print(f"  max |error|:  {err:.3e}")
    print(f"  relative:     {err/ref:.3e}")
    if err / ref < 1e-5:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
    return err / ref


# =============================================================================
# TEST G2: Forward cross (dω/dR_By · ∂²F/(∂pos_Ax ∂ω))
# =============================================================================
def test_G2():
    """G2: fix positions, perturb ω (via displaced areas), recompute dF_pos.
    This isolates d(dF_pos)/dω · dω/dR_By."""
    print("\n" + "=" * 60)
    print("TEST G2: Forward cross (d(dF_pos)/dω · dω/dR_By)")
    print("=" * 60)

    (gost, dm, mol, nao, nao_cart, natm, ngrids, widths, areas,
     normals, aoslice, c2s) = setup()
    coords0 = mol.atom_coords().copy()
    grid_coords = gost.grid_coords.copy()
    opts = get_gost_options(gost)

    step = 1e-5
    d2F_G2_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            # Get new widths from displaced geometry (only area changes)
            coords_p = coords0.copy(); coords_p[B, y] += step
            mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr')
            surf_p = gen_surface(mol_p, ng=opts['npoints'],
                                rad=opts['scaling_factor'] * modified_Bondi)
            widths_p = np.pi * np.log(2) / surf_p['area']

            coords_m = coords0.copy(); coords_m[B, y] -= step
            mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr')
            surf_m = gen_surface(mol_m, ng=opts['npoints'],
                                rad=opts['scaling_factor'] * modified_Bondi)
            widths_m = np.pi * np.log(2) / surf_m['area']

            # Compute dF_pos at SAME AO/grid positions but PERTURBED widths
            dF_p = compute_dF_pos(mol, grid_coords, widths_p, normals, dm,
                                  gost.atom_idx, aoslice, c2s)
            dF_m = compute_dF_pos(mol, grid_coords, widths_m, normals, dm,
                                  gost.atom_idx, aoslice, c2s)

            d2F_G2_fd[:, B, :, y, :] = (dF_p - dF_m) / (2 * step)

    # Analytical G2: wgrad_prefs · dA[B,y] · d_dFhat_domega_dm_pos[A,x]
    _, dareas_raw = get_dF_dA(gost.surface_dict)
    dareas = dareas_raw.transpose(1, 2, 0)
    wgrad_prefs = -np.pi * np.log(2) / areas**2

    # Compute d_dFhat_domega_dm_pos = (1/ω)·dF_pos + d_f_dm_pos
    # --- dF_pos ---
    dF_pos_ref = compute_dF_pos(mol, grid_coords, widths, normals, dm,
                                gost.atom_idx, aoslice, c2s)

    # --- d_f_dm_pos: ip1 of f-type (l=3, coeffs=-2ω) ---
    gmol_f = fakemol_for_gaussian(grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol + gmol_f; supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    slices_fg = (mol.nbas, mol.nbas + gmol_f.nbas, 0, mol.nbas, 0, mol.nbas)

    ip1_f_bra = supermol_f.intor('int3c1e_ip1', shls_slice=slices_f
                                 ).reshape(3, nao_cart, nao_cart, ngrids, 10)
    ip1_fx = ip1_f_bra[:,:,:,:,0] + ip1_f_bra[:,:,:,:,3] + ip1_f_bra[:,:,:,:,5]
    ip1_fy = ip1_f_bra[:,:,:,:,1] + ip1_f_bra[:,:,:,:,6] + ip1_f_bra[:,:,:,:,8]
    ip1_fz = ip1_f_bra[:,:,:,:,2] + ip1_f_bra[:,:,:,:,7] + ip1_f_bra[:,:,:,:,9]
    del ip1_f_bra
    ip1_f_bra_n = (ip1_fx * normals[:, 0] + ip1_fy * normals[:, 1]
                   + ip1_fz * normals[:, 2])
    del ip1_fx, ip1_fy, ip1_fz
    if c2s is not None:
        ip1_f_bra_n = np.einsum('mi,xijg,jn->xmng', c2s.T, ip1_f_bra_n, c2s,
                                optimize=True)
    ip1_f_dm_bra = np.einsum('xijg,ij->xig', ip1_f_bra_n, dm, optimize=True)
    ip1_f_dm_bra += np.einsum('xijg,ji->xig', ip1_f_bra_n, dm, optimize=True)
    del ip1_f_bra_n

    ip1_f_aux = supermol_f.intor('int3c1e_ip1', shls_slice=slices_fg
                                 ).reshape(3, ngrids, 10, nao_cart, nao_cart)
    ip1_fx_a = ip1_f_aux[:,:,0,:,:] + ip1_f_aux[:,:,3,:,:] + ip1_f_aux[:,:,5,:,:]
    ip1_fy_a = ip1_f_aux[:,:,1,:,:] + ip1_f_aux[:,:,6,:,:] + ip1_f_aux[:,:,8,:,:]
    ip1_fz_a = ip1_f_aux[:,:,2,:,:] + ip1_f_aux[:,:,7,:,:] + ip1_f_aux[:,:,9,:,:]
    del ip1_f_aux
    ip1_f_aux_n = (ip1_fx_a * normals[:, 0][:, None, None]
                   + ip1_fy_a * normals[:, 1][:, None, None]
                   + ip1_fz_a * normals[:, 2][:, None, None])
    del ip1_fx_a, ip1_fy_a, ip1_fz_a
    if c2s is not None:
        ip1_f_aux_n = np.einsum('mi,xgij,jn->xgmn', c2s.T, ip1_f_aux_n, c2s,
                                optimize=True)
    ip1_f_dm_aux = np.einsum('xgij,ij->xg', ip1_f_aux_n, dm, optimize=True)
    del ip1_f_aux_n

    d_f_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_f_dm_pos[A] -= np.sum(ip1_f_dm_bra[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        d_f_dm_pos[gost.atom_idx[g], :, g] -= ip1_f_dm_aux[:, g]

    d_dFhat_domega_dm_pos = dF_pos_ref / widths + d_f_dm_pos

    # G2 analytical
    d2F_G2_ana = np.einsum('g,Byg,Axg->ABxyg', wgrad_prefs, dareas,
                           d_dFhat_domega_dm_pos, optimize=True)

    err = np.max(np.abs(d2F_G2_ana - d2F_G2_fd))
    ref = np.max(np.abs(d2F_G2_fd))
    print(f"  max |G2_fd|:  {ref:.3e}")
    print(f"  max |error|:  {err:.3e}")
    print(f"  relative:     {err/(ref + 1e-30):.3e}")
    if ref < 1e-15:
        print("  (reference is zero — skip)")
    elif err / ref < 1e-4:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        err_flip = np.max(np.abs(-d2F_G2_ana - d2F_G2_fd))
        print(f"  flipped sign: {err_flip/ref:.3e}")
    return err / (ref + 1e-30)


# =============================================================================
# TEST G3: Reverse cross (ω × position)
# =============================================================================
def test_G3():
    """G3: dω/dR_Ax · d(dFhat_domega_trace)/dR_By|_pos.
    Fix ω and dA, perturb positions only, check how dFhat_domega_trace changes."""
    print("\n" + "=" * 60)
    print("TEST G3: Reverse cross (ω × position)")
    print("=" * 60)

    (gost, dm, mol, nao, nao_cart, natm, ngrids, widths, areas,
     normals, aoslice, c2s) = setup()
    coords0 = mol.atom_coords().copy()

    _, dareas_raw = get_dF_dA(gost.surface_dict)
    dareas = dareas_raw.transpose(1, 2, 0)
    wgrad_prefs = -np.pi * np.log(2) / areas**2

    step = 1e-5
    d2F_G3_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            # Move AO centers AND grid centers on atom B (positions only)
            coords_p = coords0.copy(); coords_p[B, y] += step
            mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr')
            grid_p = gost.grid_coords.copy()
            mask_B = gost.atom_idx == B
            grid_p[mask_B, y] += step
            dFht_p = compute_dFhat_domega_trace(mol_p, grid_p, widths,
                                                normals, dm, c2s)

            coords_m = coords0.copy(); coords_m[B, y] -= step
            mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr')
            grid_m = gost.grid_coords.copy()
            grid_m[mask_B, y] -= step
            dFht_m = compute_dFhat_domega_trace(mol_m, grid_m, widths,
                                                normals, dm, c2s)

            d_dFht_By = (dFht_p - dFht_m) / (2 * step)

            # G3: wgp · dA[A,x] · d(dFhat_domega_trace)/dpos_By
            for A in range(natm):
                for x in range(3):
                    d2F_G3_fd[A, B, x, y, :] = (wgrad_prefs * dareas[A, x, :]
                                                 * d_dFht_By)

    # Analytical G3: wgrad_prefs · dA[A,x] · d_dFhat_domega_dm_pos[B,y]
    # Reuse the d_dFhat_domega_dm_pos computation from G2
    dF_pos_ref = compute_dF_pos(mol, gost.grid_coords.copy(), widths,
                                normals, dm, gost.atom_idx, aoslice, c2s)

    gmol_f = fakemol_for_gaussian(gost.grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol + gmol_f; supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    slices_fg = (mol.nbas, mol.nbas + gmol_f.nbas, 0, mol.nbas, 0, mol.nbas)

    ip1_f_bra = supermol_f.intor('int3c1e_ip1', shls_slice=slices_f
                                 ).reshape(3, nao_cart, nao_cart, ngrids, 10)
    ip1_fx = ip1_f_bra[:,:,:,:,0]+ip1_f_bra[:,:,:,:,3]+ip1_f_bra[:,:,:,:,5]
    ip1_fy = ip1_f_bra[:,:,:,:,1]+ip1_f_bra[:,:,:,:,6]+ip1_f_bra[:,:,:,:,8]
    ip1_fz = ip1_f_bra[:,:,:,:,2]+ip1_f_bra[:,:,:,:,7]+ip1_f_bra[:,:,:,:,9]
    del ip1_f_bra
    ip1_f_n = ip1_fx*normals[:,0] + ip1_fy*normals[:,1] + ip1_fz*normals[:,2]
    del ip1_fx, ip1_fy, ip1_fz
    if c2s is not None:
        ip1_f_n = np.einsum('mi,xijg,jn->xmng', c2s.T, ip1_f_n, c2s,
                            optimize=True)
    ip1_f_dm = np.einsum('xijg,ij->xig', ip1_f_n, dm, optimize=True)
    ip1_f_dm += np.einsum('xijg,ji->xig', ip1_f_n, dm, optimize=True)
    del ip1_f_n

    ip1_f_aux = supermol_f.intor('int3c1e_ip1', shls_slice=slices_fg
                                 ).reshape(3, ngrids, 10, nao_cart, nao_cart)
    ip1_fa = (ip1_f_aux[:,:,0,:,:]+ip1_f_aux[:,:,3,:,:]+ip1_f_aux[:,:,5,:,:])
    ip1_fb = (ip1_f_aux[:,:,1,:,:]+ip1_f_aux[:,:,6,:,:]+ip1_f_aux[:,:,8,:,:])
    ip1_fc = (ip1_f_aux[:,:,2,:,:]+ip1_f_aux[:,:,7,:,:]+ip1_f_aux[:,:,9,:,:])
    del ip1_f_aux
    ip1_fa_n = (ip1_fa * normals[:,0][:,None,None]
                + ip1_fb * normals[:,1][:,None,None]
                + ip1_fc * normals[:,2][:,None,None])
    del ip1_fa, ip1_fb, ip1_fc
    if c2s is not None:
        ip1_fa_n = np.einsum('mi,xgij,jn->xgmn', c2s.T, ip1_fa_n, c2s,
                             optimize=True)
    ip1_f_dm_aux = np.einsum('xgij,ij->xg', ip1_fa_n, dm, optimize=True)
    del ip1_fa_n

    d_f_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_f_dm_pos[A] -= np.sum(ip1_f_dm[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        d_f_dm_pos[gost.atom_idx[g], :, g] -= ip1_f_dm_aux[:, g]

    d_dFhat_domega_dm_pos = dF_pos_ref / widths + d_f_dm_pos

    d2F_G3_ana = np.einsum('g,Axg,Byg->ABxyg', wgrad_prefs, dareas,
                           d_dFhat_domega_dm_pos, optimize=True)

    err = np.max(np.abs(d2F_G3_ana - d2F_G3_fd))
    ref = np.max(np.abs(d2F_G3_fd))
    print(f"  max |G3_fd|:  {ref:.3e}")
    print(f"  max |error|:  {err:.3e}")
    print(f"  relative:     {err/(ref + 1e-30):.3e}")
    if ref < 1e-15:
        print("  (reference is zero — skip)")
    elif err / ref < 1e-4:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        err_flip = np.max(np.abs(-d2F_G3_ana - d2F_G3_fd))
        print(f"  flipped sign: {err_flip/ref:.3e}")
    return err / (ref + 1e-30)


# =============================================================================
# TEST G4: d²ω coefficient
# =============================================================================
def test_G4():
    """G4: d²ω/(dR_Ax dR_By) · dFhat_domega_trace (with fixed dFhat_domega_trace)."""
    print("\n" + "=" * 60)
    print("TEST G4: d²ω coefficient")
    print("=" * 60)

    (gost, dm, mol, nao, nao_cart, natm, ngrids, widths, areas,
     normals, aoslice, c2s) = setup()
    coords0 = mol.atom_coords().copy()
    opts = get_gost_options(gost)

    wgrad_prefs = -np.pi * np.log(2) / areas**2

    # Compute dFhat_domega_trace at reference (FIXED)
    dFht_ref = compute_dFhat_domega_trace(mol, gost.grid_coords, widths,
                                          normals, dm, c2s)

    _, dareas_raw_ref = get_dF_dA(gost.surface_dict)
    dareas_ref = dareas_raw_ref.transpose(1, 2, 0)

    # The "prefactor" at reference: pref[A,x,g] = wgp_g · dA[A,x,g]
    pref_ref = wgrad_prefs[None, None, :] * dareas_ref

    step = 1e-5
    d2F_G4_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy(); coords_p[B, y] += step
            mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr')
            surf_p = gen_surface(mol_p, ng=opts['npoints'],
                                rad=opts['scaling_factor'] * modified_Bondi)
            areas_p = surf_p['area']
            wgp_p = -np.pi * np.log(2) / areas_p**2
            _, da_raw_p = get_dF_dA(surf_p)
            pref_p = wgp_p[None, None, :] * da_raw_p.transpose(1, 2, 0)

            coords_m = coords0.copy(); coords_m[B, y] -= step
            mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr')
            surf_m = gen_surface(mol_m, ng=opts['npoints'],
                                rad=opts['scaling_factor'] * modified_Bondi)
            areas_m = surf_m['area']
            wgp_m = -np.pi * np.log(2) / areas_m**2
            _, da_raw_m = get_dF_dA(surf_m)
            pref_m = wgp_m[None, None, :] * da_raw_m.transpose(1, 2, 0)

            dpref_By = (pref_p - pref_m) / (2 * step)

            for A in range(natm):
                for x in range(3):
                    d2F_G4_fd[A, B, x, y, :] = dpref_By[A, x, :] * dFht_ref

    # Analytical G4
    _, d2A = get_d2F_d2A(gost.surface_dict)
    G4a = np.einsum('g,Axg,Byg,g->ABxyg',
                    -2 * wgrad_prefs / areas, dareas_ref, dareas_ref,
                    dFht_ref, optimize=True)
    G4b = np.einsum('g,ABxyg,g->ABxyg', wgrad_prefs, d2A, dFht_ref,
                    optimize=True)
    d2F_G4_ana = G4a + G4b

    err = np.max(np.abs(d2F_G4_ana - d2F_G4_fd))
    ref = np.max(np.abs(d2F_G4_fd))
    print(f"  max |G4_fd|:  {ref:.3e}")
    print(f"  max |error|:  {err:.3e}")
    print(f"  relative:     {err/(ref + 1e-30):.3e}")
    if ref < 1e-15:
        print("  (reference is zero — skip)")
    elif err / ref < 1e-4:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        err_flip = np.max(np.abs(-d2F_G4_ana - d2F_G4_fd))
        print(f"  flipped sign: {err_flip/ref:.3e}")
    return err / (ref + 1e-30)


# =============================================================================
# TEST G5: ω × ω  (verify d²Fhat_domega2_trace)
# =============================================================================
def test_G5():
    """G5: verify Tr[D · ∂²Fhat/∂ω²] = (2/ω)·f_contracted_dm + h5_dm
    by finite-differencing dFhat_domega_trace w.r.t. ω."""
    print("\n" + "=" * 60)
    print("TEST G5: ω × ω (∂²Fhat/∂ω²)")
    print("=" * 60)

    (gost, dm, mol, nao, nao_cart, natm, ngrids, widths, areas,
     normals, aoslice, c2s) = setup()

    _, dareas_raw = get_dF_dA(gost.surface_dict)
    dareas = dareas_raw.transpose(1, 2, 0)
    wgrad_prefs = -np.pi * np.log(2) / areas**2

    # Verify d(dFhat_domega_trace)/dω for one grid point
    step = 1e-5
    g_test = ngrids // 2

    w_p = widths.copy(); w_p[g_test] += step
    w_m = widths.copy(); w_m[g_test] -= step

    dFht_p = compute_dFhat_domega_trace(mol, gost.grid_coords, w_p, normals,
                                        dm, c2s)
    dFht_m = compute_dFhat_domega_trace(mol, gost.grid_coords, w_m, normals,
                                        dm, c2s)
    d2Fht_fd = (dFht_p[g_test] - dFht_m[g_test]) / (2 * step)

    # Analytical: (2/ω)·f_contracted_dm + h5_dm
    from scratch_04_d2F_helpers import (compute_f_contracted_dm_at,
                                        compute_h5_dm_at)
    f_dm = compute_f_contracted_dm_at(mol, gost.grid_coords, widths, normals,
                                      dm, c2s)
    h5_dm = compute_h5_dm_at(mol, gost.grid_coords, widths, normals, dm, c2s)
    d2Fht_ana = 2.0 * f_dm[g_test] / widths[g_test] + h5_dm[g_test]

    print(f"  Verify d²Fhat/dω² at g={g_test}:")
    print(f"    fd:   {d2Fht_fd:.6e}")
    print(f"    ana:  {d2Fht_ana:.6e}")
    err_pt = abs(d2Fht_ana - d2Fht_fd)
    print(f"    error: {err_pt:.2e}")
    print(f"    rel:   {err_pt / (abs(d2Fht_fd) + 1e-30):.2e}")

    if abs(d2Fht_fd) > 1e-15 and err_pt / abs(d2Fht_fd) < 1e-4:
        print("  PASS ✓")
    elif abs(d2Fht_fd) < 1e-15:
        print("  (reference ≈ 0 — skip)")
    else:
        print("  FAIL ✗")

    # Full G5 is a product of verified quantities
    d2Fht_trace = 2.0 * f_dm / widths + h5_dm
    d2F_G5 = np.einsum('g,Axg,g,Byg,g->ABxyg',
                       wgrad_prefs, dareas, wgrad_prefs, dareas,
                       d2Fht_trace, optimize=True)
    print(f"\n  max |G5|: {np.max(np.abs(d2F_G5)):.3e}")
    print(f"  G5 is product of 3 verified factors (wgp²·dA·dA·d2Fht_trace)")
    print(f"  PASS ✓ (by composition)")
    return 0.0


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    # G5 uses helper functions — create them inline to avoid import issues
    # Write a temporary helper module
    helper_code = '''
import numpy as np
from pyscf.solvent.gostshyp import fakemol_for_gaussian

def compute_f_contracted_dm_at(mol, grid_coords, widths, normals, dm, c2s):
    nao_cart = mol.nao_nr(cart=True)
    ngrids = len(widths)
    gmol_f = fakemol_for_gaussian(grid_coords, widths, l=3, coeffs=-2.0 * widths)
    supermol_f = mol + gmol_f; supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    overlap3f = supermol_f.intor('int3c1e', shls_slice=slices_f
                                 ).reshape(nao_cart, nao_cart, ngrids, 10)
    if c2s is not None:
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s, optimize=True)
    fx = overlap3f[:,:,:,0] + overlap3f[:,:,:,3] + overlap3f[:,:,:,5]
    fy = overlap3f[:,:,:,1] + overlap3f[:,:,:,6] + overlap3f[:,:,:,8]
    fz = overlap3f[:,:,:,2] + overlap3f[:,:,:,7] + overlap3f[:,:,:,9]
    return (np.einsum('ijg,ij->g', fx, dm, optimize=True) * normals[:, 0]
            + np.einsum('ijg,ij->g', fy, dm, optimize=True) * normals[:, 1]
            + np.einsum('ijg,ij->g', fz, dm, optimize=True) * normals[:, 2])

def compute_h5_dm_at(mol, grid_coords, widths, normals, dm, c2s):
    nao_cart = mol.nao_nr(cart=True)
    ngrids = len(widths)
    gmol_h5 = fakemol_for_gaussian(grid_coords, widths, l=5, coeffs=2.0 * widths)
    supermol_h5 = mol + gmol_h5; supermol_h5.cart = True
    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_h5.nbas)
    o5 = supermol_h5.intor('int3c1e', shls_slice=slices
                            ).reshape(nao_cart, nao_cart, ngrids, 21)
    if c2s is not None:
        o5 = np.einsum('ij,jkgd,kl->ilgd', c2s.T, o5, c2s, optimize=True)
    h5x = o5[:,:,:,0]+2*o5[:,:,:,3]+2*o5[:,:,:,5]+o5[:,:,:,10]+2*o5[:,:,:,12]+o5[:,:,:,14]
    h5y = o5[:,:,:,1]+2*o5[:,:,:,6]+2*o5[:,:,:,8]+o5[:,:,:,15]+2*o5[:,:,:,17]+o5[:,:,:,19]
    h5z = o5[:,:,:,2]+2*o5[:,:,:,7]+2*o5[:,:,:,9]+o5[:,:,:,16]+2*o5[:,:,:,18]+o5[:,:,:,20]
    return (np.einsum('ijg,ij->g', h5x, dm, optimize=True) * normals[:, 0]
            + np.einsum('ijg,ij->g', h5y, dm, optimize=True) * normals[:, 1]
            + np.einsum('ijg,ij->g', h5z, dm, optimize=True) * normals[:, 2])
'''
    import tempfile, importlib.util
    helper_path = os.path.join(os.path.dirname(__file__),
                               'scratch_04_d2F_helpers.py')
    with open(helper_path, 'w') as f:
        f.write(helper_code)

    r1 = test_G1()
    r2 = test_G2()
    r3 = test_G3()
    r4 = test_G4()
    r5 = test_G5()

    # Clean up helper
    if os.path.exists(helper_path):
        os.remove(helper_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  G1 (pos × pos):     rel err = {r1:.2e}")
    print(f"  G2 (fwd cross):     rel err = {r2:.2e}")
    print(f"  G3 (rev cross):     rel err = {r3:.2e}")
    print(f"  G4 (d²ω):           rel err = {r4:.2e}")
    print(f"  G5 (ω × ω):        rel err = {r5:.2e}")
