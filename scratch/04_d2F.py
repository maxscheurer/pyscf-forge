#!/usr/bin/env python
"""
Test 4: Validate d²F_g/(dR_Ax dR_By) — second derivative of Fhat trace.

This is needed for Hessian term H9: -A·e/F² · d²F/(dR_Ax dR_By)

Same 5-group structure as d²e but with p-type fakemol contracted with normals.
Key differences from d²e:
  - Base integral is p-type (l=1) with coeffs=2ω, contracted with surface normals
  - ∂F/∂ω = F/ω + f_contracted (f-type l=3, coeffs=-2ω), NOT just -diagd
  - ∂²F/∂ω² = (2/ω)·f_contracted + h5_contracted (l=5, coeffs=2ω)
  - Cross terms use ip1 of f-type + (1/ω)·dF_pos (not just ip1 of d-type)

Groups:
  G1: Position × Position (ipip1, ipvip1, ip1ip2, ipip2)
  G2: Forward cross: dω/dR_By · ∂²F/(∂pos_Ax ∂ω)
  G3: Reverse cross: dω/dR_Ax · ∂²F/(∂pos_By ∂ω)
  G4: d²ω/(dR_Ax dR_By) · ∂F/∂ω  (uses dFhat_domega_trace)
  G5: dω/dR_Ax · dω/dR_By · ∂²F/∂ω²  (uses l=5 + (2/ω)·f_contracted_dm)
"""

import numpy as np
from pyscf import gto
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A

from common import (make_h2_system, make_h2o_system, compute_scalar_traces,
                    get_gost_options)


def compute_d2F_fd(gost, dm, mol, step=1e-4):
    """Compute d²F_g/(dR_Ax dR_By) by finite-differencing dF_g/dR.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    natm = mol.natm
    ngrids = gost.n_gaussian
    coords0 = mol.atom_coords().copy()
    opts = get_gost_options(gost)

    d2F_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy()
            coords_p[B, y] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=opts)
            gost_p.kernel(dm)
            _, dF_p = compute_scalar_traces(gost_p, dm)

            coords_m = coords0.copy()
            coords_m[B, y] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=opts)
            gost_m.kernel(dm)
            _, dF_m = compute_scalar_traces(gost_m, dm)

            d2F_fd[:, B, :, y, :] = (dF_p - dF_m) / (2.0 * step)

    return d2F_fd


def compute_f_contracted_dm(gost, dm, mol):
    """Compute f_contracted_dm = Tr[D · f_contracted] (f-type l=3 with normals).

    Returns scalar per grid point, shape (ngrids,).
    """
    mol_obj = gost.mol
    nao_cart = mol_obj.nao_nr(cart=True)
    ngrids = gost.n_gaussian
    widths = gost.widths
    normals = gost.surface_normals

    if not mol_obj.cart:
        c2s = mol_obj.cart2sph_coeff(normalized='sp')

    gmol_f = fakemol_for_gaussian(gost.grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol_obj + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_f.nbas)
    overlap3f = supermol_f.intor('int3c1e', shls_slice=slices_f
                                 ).reshape(nao_cart, nao_cart, ngrids, 10)
    if not mol_obj.cart:
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s,
                              optimize=True)

    # Laplacian-trace per Cartesian direction: r_c · r²
    # l=3 ordering: xxx(0), xxy(1), xxz(2), xyy(3), xyz(4), xzz(5),
    #               yyy(6), yyz(7), yzz(8), zzz(9)
    fx = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    fy = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    fz = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]

    f_contracted_dm = (
        np.einsum('ijg,ij->g', fx, dm, optimize=True) * normals[:, 0]
        + np.einsum('ijg,ij->g', fy, dm, optimize=True) * normals[:, 1]
        + np.einsum('ijg,ij->g', fz, dm, optimize=True) * normals[:, 2]
    )
    return f_contracted_dm


def compute_h5_contracted_dm(gost, dm, mol):
    """Compute h5_contracted_dm = Tr[D · h5_contracted] (l=5 with normals).

    h5 comes from ∂(f_contracted)/∂ω extra piece: 2ω · r_c · r⁴ · exp(-ω r²).
    Returns scalar per grid point, shape (ngrids,).
    """
    mol_obj = gost.mol
    nao_cart = mol_obj.nao_nr(cart=True)
    ngrids = gost.n_gaussian
    widths = gost.widths
    normals = gost.surface_normals

    if not mol_obj.cart:
        c2s = mol_obj.cart2sph_coeff(normalized='sp')

    gmol_h5 = fakemol_for_gaussian(gost.grid_coords, widths, l=5,
                                   coeffs=2.0 * widths)
    supermol_h5 = mol_obj + gmol_h5
    supermol_h5.cart = True
    slices_h5 = (0, mol_obj.nbas, 0, mol_obj.nbas,
                 mol_obj.nbas, mol_obj.nbas + gmol_h5.nbas)
    overlap3h = supermol_h5.intor('int3c1e', shls_slice=slices_h5
                                  ).reshape(nao_cart, nao_cart, ngrids, 21)
    if not mol_obj.cart:
        overlap3h = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3h, c2s,
                              optimize=True)

    # l=5 lexicographic ordering (21 components):
    # 0:x5, 1:x4y, 2:x4z, 3:x3y2, 4:x3yz, 5:x3z2,
    # 6:x2y3, 7:x2y2z, 8:x2yz2, 9:x2z3, 10:xy4, 11:xy3z,
    # 12:xy2z2, 13:xyz3, 14:xz4, 15:y5, 16:y4z, 17:y3z2,
    # 18:y2z3, 19:yz4, 20:z5
    #
    # r_c · r⁴ = r_c · (x²+y²+z²)²:
    # For c=x: x⁵ + 2x³y² + 2x³z² + xy⁴ + 2xy²z² + xz⁴
    h5x = (overlap3h[:, :, :, 0] + 2*overlap3h[:, :, :, 3]
            + 2*overlap3h[:, :, :, 5] + overlap3h[:, :, :, 10]
            + 2*overlap3h[:, :, :, 12] + overlap3h[:, :, :, 14])
    # For c=y: x⁴y + 2x²y³ + 2x²yz² + y⁵ + 2y³z² + yz⁴
    h5y = (overlap3h[:, :, :, 1] + 2*overlap3h[:, :, :, 6]
            + 2*overlap3h[:, :, :, 8] + overlap3h[:, :, :, 15]
            + 2*overlap3h[:, :, :, 17] + overlap3h[:, :, :, 19])
    # For c=z: x⁴z + 2x²y²z + 2x²z³ + y⁴z + 2y²z³ + z⁵
    h5z = (overlap3h[:, :, :, 2] + 2*overlap3h[:, :, :, 7]
            + 2*overlap3h[:, :, :, 9] + overlap3h[:, :, :, 16]
            + 2*overlap3h[:, :, :, 18] + overlap3h[:, :, :, 20])

    # Contract with normals and trace with D
    h5_dm = (
        np.einsum('ijg,ij->g', h5x, dm, optimize=True) * normals[:, 0]
        + np.einsum('ijg,ij->g', h5y, dm, optimize=True) * normals[:, 1]
        + np.einsum('ijg,ij->g', h5z, dm, optimize=True) * normals[:, 2]
    )
    return h5_dm


def compute_d2F_analytical(gost, dm, mol):
    """Compute d²F_g/(dR_Ax dR_By) analytically — ALL terms.

    Returns shape (natm, natm, 3, 3, ngrids).

    Decomposition:
      d²F/(dR_Ax dR_By) =
        (G1) ∂²F/(∂pos_Ax ∂pos_By)                    — position × position
        (G2) dω/dR_By · ∂²F/(∂pos_Ax ∂ω)             — forward cross
        (G3) dω/dR_Ax · ∂²F/(∂pos_By ∂ω)             — reverse cross
        (G4) d²ω/(dR_Ax dR_By) · ∂F/∂ω               — d²ω piece
        (G5) dω/dR_Ax · dω/dR_By · ∂²F/∂ω²           — ω×ω piece
    """
    mol_obj = gost.mol
    nao = mol_obj.nao_nr()
    nao_cart = mol_obj.nao_nr(cart=True)
    natm = mol_obj.natm
    ngrids = gost.n_gaussian
    aoslice = mol_obj.aoslice_by_atom()

    widths = gost.widths
    areas = gost.areas
    normals = gost.surface_normals
    forces = gost.forces
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)  # dω/dA

    if not mol_obj.cart:
        c2s = mol_obj.cart2sph_coeff(normalized='sp')

    # Area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0)  # (natm, 3, ngrids)
        _, d2A = get_d2F_d2A(gost.surface_dict)

    d2F = np.zeros((natm, natm, 3, 3, ngrids))

    # =====================================================================
    # PART 1 (G1): Position × Position (second-derivative integrals)
    # =====================================================================
    # Sign convention: ipip integrals give ∂²/∂r² (electron coords).
    # Two sign flips cancel: ∂²/∂R² = (-1)(-1)∂²/∂r² = +∂²/∂r² → use +=
    # This is identical to d²e, just with p-type aux shell + normal contraction.
    # =====================================================================
    gmol_p = fakemol_for_gaussian(gost.grid_coords, widths, l=1,
                                  coeffs=2.0 * widths)
    supermol = mol_obj + gmol_p
    slices_p = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_p.nbas)

    # --- ipip1: ∂²/∂R_bra² (A=B diagonal) ---
    ipip1_raw = supermol.intor('int3c1e_ipip1', shls_slice=slices_p
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip1 = np.einsum('xijgc,gc->xijg', ipip1_raw, normals, optimize=True)
    del ipip1_raw

    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib = np.sum(ipip1_dm[:, p0:p1, :], axis=1)
        d2F[A, A, :, :, :] += contrib.reshape(3, 3, ngrids)
    del ipip1, ipip1_dm

    # --- ipvip1: ∂²/(∂R_bra ∂R_ket) ---
    ipvip1_raw = supermol.intor('int3c1e_ipvip1', shls_slice=slices_p
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
            d2F[A, B, :, :, :] += trace_block.reshape(3, 3, ngrids)
    del ipvip1

    # --- ip1ip2: ∂²/(∂R_bra ∂R_aux) ---
    ip1ip2_raw = supermol.intor('int3c1e_ip1ip2', shls_slice=slices_p
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
            d2F[A, B, :, :, g] += contrib_A[:, g].reshape(3, 3)

    # --- aux × bra/ket (reverse of ip1ip2) ---
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            trace_ig = np.sum(ip1ip2_dm[:, p0:p1, g], axis=1)
            d2F[A, B, :, :, g] += trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm

    # --- ipip2: ∂²/∂R_aux² (both on aux center, same atom) ---
    ipip2_raw = supermol.intor('int3c1e_ipip2', shls_slice=slices_p
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip2 = np.einsum('xijgc,gc->xijg', ipip2_raw, normals, optimize=True)
    del ipip2_raw

    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2F[B, B, :, :, g] += ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    # =====================================================================
    # PART 2 (G2+G3): Position × Width cross terms
    # =====================================================================
    # ∂²F/(∂pos_Ax ∂ω) = (1/ω) · ∂F/∂pos_Ax + ∂(f_contracted)/∂pos_Ax
    #
    # where ∂F/∂pos is from ip1 of p-type (normal contracted),
    # and ∂(f_contracted)/∂pos is from ip1 of f-type (l=3, coeffs=-2ω,
    # Laplacian-traced, normal contracted).
    #
    # Trace with D:
    #   Tr[D · ∂²F/(∂pos_Ax ∂ω)] = (1/ω)·dF_pos[A,x] + d_f_dm_pos[A,x]
    #                               =: d_dFhat_domega_dm_pos[A,x]
    #
    # G2: dω/dR_By · d_dFhat_domega_dm_pos[A,x]
    # G3: dω/dR_Ax · d_dFhat_domega_dm_pos[B,y]
    # =====================================================================

    # --- Compute dF_pos: position-only part of dF/dR ---
    # (same p-type ip1 as in common.py's compute_scalar_traces)
    slices_pg = (mol_obj.nbas, mol_obj.nbas + gmol_p.nbas,
                 0, mol_obj.nbas, 0, mol_obj.nbas)

    # Bra/ket ip1 of p-type
    ip1_p_bra_raw = supermol.intor('int3c1e_ip1', shls_slice=slices_p
                                   ).reshape(3, nao, nao, ngrids, 3)
    ip1_p_bra = np.einsum('xijgc,gc->xijg', ip1_p_bra_raw, normals,
                          optimize=True)
    del ip1_p_bra_raw

    ip1_p_bra_dm = np.einsum('xijg,ij->xig', ip1_p_bra, dm, optimize=True)
    ip1_p_bra_dm += np.einsum('xijg,ji->xig', ip1_p_bra, dm, optimize=True)
    del ip1_p_bra

    # Aux ip1 of p-type
    ip1_p_aux_raw = supermol.intor('int3c1e_ip1', shls_slice=slices_pg
                                   ).reshape(3, ngrids, 3, nao, nao)
    ip1_p_aux = np.einsum('xgcij,gc->xgij', ip1_p_aux_raw, normals,
                          optimize=True)
    del ip1_p_aux_raw
    ip1_p_aux_dm = np.einsum('xgij,ij->xg', ip1_p_aux, dm, optimize=True)
    del ip1_p_aux

    # Build dF_pos (sign: ip1 = ∂/∂r, so ∂F/∂R = -ip1 → use -=)
    dF_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dF_pos[A, :, :] -= np.sum(ip1_p_bra_dm[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        dF_pos[gost.atom_idx[g], :, g] -= ip1_p_aux_dm[:, g]
    del ip1_p_bra_dm, ip1_p_aux_dm

    # --- Compute d_f_dm_pos: position derivative of f_contracted_dm ---
    # f-type (l=3, coeffs=-2ω)
    gmol_f = fakemol_for_gaussian(gost.grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol_obj + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_f.nbas)
    slices_fg = (mol_obj.nbas, mol_obj.nbas + gmol_f.nbas,
                 0, mol_obj.nbas, 0, mol_obj.nbas)

    # Bra ip1 of f-type: shape (3, nao_cart, nao_cart, ngrids*10)
    ip1_f_bra = supermol_f.intor('int3c1e_ip1', shls_slice=slices_f
                                 ).reshape(3, nao_cart, nao_cart, ngrids, 10)
    # Laplacian-trace per direction, then contract with normals
    ip1_fx = ip1_f_bra[:, :, :, :, 0] + ip1_f_bra[:, :, :, :, 3] + ip1_f_bra[:, :, :, :, 5]
    ip1_fy = ip1_f_bra[:, :, :, :, 1] + ip1_f_bra[:, :, :, :, 6] + ip1_f_bra[:, :, :, :, 8]
    ip1_fz = ip1_f_bra[:, :, :, :, 2] + ip1_f_bra[:, :, :, :, 7] + ip1_f_bra[:, :, :, :, 9]
    del ip1_f_bra
    # Contract with normals: shape (3, nao_cart, nao_cart, ngrids)
    ip1_f_contracted_bra = (ip1_fx * normals[:, 0]
                            + ip1_fy * normals[:, 1]
                            + ip1_fz * normals[:, 2])
    del ip1_fx, ip1_fy, ip1_fz
    # Cart-to-sph
    if not mol_obj.cart:
        ip1_f_contracted_bra = np.einsum('mi,xijg,jn->xmng', c2s.T,
                                         ip1_f_contracted_bra, c2s,
                                         optimize=True)
    # Trace with D (bra+ket since ip1 breaks μν symmetry)
    ip1_f_dm_bra = np.einsum('xijg,ij->xig', ip1_f_contracted_bra, dm,
                             optimize=True)
    ip1_f_dm_bra += np.einsum('xijg,ji->xig', ip1_f_contracted_bra, dm,
                              optimize=True)
    del ip1_f_contracted_bra

    # Aux ip1 of f-type: shape (3, ngrids*10, nao_cart, nao_cart)
    ip1_f_aux = supermol_f.intor('int3c1e_ip1', shls_slice=slices_fg
                                 ).reshape(3, ngrids, 10, nao_cart, nao_cart)
    ip1_fx_aux = ip1_f_aux[:, :, 0, :, :] + ip1_f_aux[:, :, 3, :, :] + ip1_f_aux[:, :, 5, :, :]
    ip1_fy_aux = ip1_f_aux[:, :, 1, :, :] + ip1_f_aux[:, :, 6, :, :] + ip1_f_aux[:, :, 8, :, :]
    ip1_fz_aux = ip1_f_aux[:, :, 2, :, :] + ip1_f_aux[:, :, 7, :, :] + ip1_f_aux[:, :, 9, :, :]
    del ip1_f_aux
    # Contract with normals: shape (3, ngrids, nao_cart, nao_cart)
    ip1_f_contracted_aux = (ip1_fx_aux * normals[:, 0][:, None, None]
                            + ip1_fy_aux * normals[:, 1][:, None, None]
                            + ip1_fz_aux * normals[:, 2][:, None, None])
    del ip1_fx_aux, ip1_fy_aux, ip1_fz_aux
    if not mol_obj.cart:
        ip1_f_contracted_aux = np.einsum('mi,xgij,jn->xgmn', c2s.T,
                                         ip1_f_contracted_aux, c2s,
                                         optimize=True)
    # Trace with D (aux deriv keeps μν symmetry → just D)
    ip1_f_dm_aux = np.einsum('xgij,ij->xg', ip1_f_contracted_aux, dm,
                             optimize=True)
    del ip1_f_contracted_aux

    # Build d_f_dm_pos (same sign convention as dF_pos: -= for ip1)
    d_f_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_f_dm_pos[A, :, :] -= np.sum(ip1_f_dm_bra[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        d_f_dm_pos[gost.atom_idx[g], :, g] -= ip1_f_dm_aux[:, g]
    del ip1_f_dm_bra, ip1_f_dm_aux

    # Combine: d_dFhat_domega_dm_pos = (1/ω) · dF_pos + d_f_dm_pos
    d_dFhat_domega_dm_pos = dF_pos / widths + d_f_dm_pos

    # G2: dω/dR_By · Tr[D · ∂²F/(∂pos_Ax ∂ω)]
    # = wgrad_prefs · dA[B,y] · d_dFhat_domega_dm_pos[A,x]
    d2F += np.einsum('g,Byg,Axg->ABxyg', wgrad_prefs, dareas,
                     d_dFhat_domega_dm_pos, optimize=True)

    # G3: dω/dR_Ax · Tr[D · ∂²F/(∂pos_By ∂ω)]  (same formula, swapped)
    d2F += np.einsum('g,Axg,Byg->ABxyg', wgrad_prefs, dareas,
                     d_dFhat_domega_dm_pos, optimize=True)

    # =====================================================================
    # PART 3 (G4): d²ω/(dR_Ax dR_By) · Tr[D · ∂Fhat/∂ω]
    # =====================================================================
    # ∂Fhat/∂ω traced with D: dFhat_domega_trace = F/ω + f_contracted_dm
    # d²ω = (-2wgp/A · dA_By · dA_Ax + wgp · d²A)
    # Contribution: d²ω · dFhat_domega_trace
    # =====================================================================
    f_contracted_dm = compute_f_contracted_dm(gost, dm, mol)
    dFhat_domega_trace = forces / widths + f_contracted_dm

    # Sub-term G4a: (-2·wgp/A · dA[A,x] · dA[B,y]) · dFhat_domega_trace
    d2F += np.einsum('g,Axg,Byg,g->ABxyg',
                     -2.0 * wgrad_prefs / areas, dareas, dareas,
                     dFhat_domega_trace, optimize=True)

    # Sub-term G4b: (wgp · d²A) · dFhat_domega_trace
    d2F += np.einsum('g,ABxyg,g->ABxyg', wgrad_prefs, d2A,
                     dFhat_domega_trace, optimize=True)

    # =====================================================================
    # PART 4 (G5): dω/dR_Ax · dω/dR_By · Tr[D · ∂²Fhat/∂ω²]
    # =====================================================================
    # ∂²Fhat/∂ω² = (2/ω) · f_contracted + h5_contracted  (l=5 piece)
    # Trace with D: (2/ω) · f_contracted_dm + h5_dm
    # =====================================================================
    h5_dm = compute_h5_contracted_dm(gost, dm, mol)
    d2Fhat_domega2_trace = 2.0 * f_contracted_dm / widths + h5_dm

    # G5: wgp·dA[A,x] · wgp·dA[B,y] · d2Fhat_domega2_trace
    d2F += np.einsum('g,Axg,g,Byg,g->ABxyg',
                     wgrad_prefs, dareas, wgrad_prefs, dareas,
                     d2Fhat_domega2_trace, optimize=True)

    return d2F


def test_d2F(gost, dm, mol, step=1e-4, label=""):
    """Test d²F_g/(dR dR) against finite differences."""
    print(f"Computing d²F_g fdiff ({label})...")
    d2F_fd = compute_d2F_fd(gost, dm, mol, step=step)

    print(f"Computing d²F_g analytical ({label})...")
    d2F_ana = compute_d2F_analytical(gost, dm, mol)

    natm = mol.natm
    ngrids = gost.n_gaussian

    err_full = np.max(np.abs(d2F_ana - d2F_fd))
    max_val = np.max(np.abs(d2F_fd))

    print(f"\n=== d²F_g Validation ({label}) ===")
    print(f"  shape: ({natm}, {natm}, 3, 3, {ngrids})")
    print(f"  max |d2F_fd|:  {max_val:.2e}")
    print(f"  max error: {err_full:.2e}")
    print(f"  relative error: {err_full/max_val:.2e}")

    if err_full / max_val < 1e-5:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        residual = d2F_fd - d2F_ana
        for A in range(natm):
            for B in range(natm):
                r = np.max(np.abs(residual[A, B]))
                if r > 1e-8:
                    print(f"    residual[{A},{B}]: {r:.2e}")

    sym_err = np.max(np.abs(d2F_fd - d2F_fd.transpose(1, 0, 3, 2, 4)))
    print(f"  fdiff symmetry: {sym_err:.2e}")

    sym_err_ana = np.max(np.abs(d2F_ana - d2F_ana.transpose(1, 0, 3, 2, 4)))
    print(f"  analytical symmetry: {sym_err_ana:.2e}")

    return d2F_ana, d2F_fd


if __name__ == '__main__':
    print("=" * 60)
    print("H2/sto-3g")
    print("=" * 60)
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")
    test_d2F(gost, dm, mol, label="H2/sto-3g")

    print("\n" + "=" * 60)
    print("H2O/cc-pVDZ")
    print("=" * 60)
    gost, dm, mol = make_h2o_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")
    test_d2F(gost, dm, mol, label="H2O/cc-pVDZ")
