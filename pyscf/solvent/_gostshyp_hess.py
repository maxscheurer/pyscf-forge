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
Analytical Hessian for the GOSTSHYP pressure model.

Computes d²E_GOSTSHYP/(dR_A dR_B) at fixed density matrix (the direct term).
The GOSTSHYP energy per grid point is:  E_g = P · A_g · e_g / F_g,
so the Hessian involves derivatives of areas (A), the Gtilde trace (e),
and the force trace (F) up to second order.

The assembly uses 10 terms (H1–H10) from the product rule applied to A·e/F.
Each of d²e and d²F decomposes into 5 groups:
  G1: position × position (ipip1, ipvip1, ip1ip2, ipip2 integrals)
  G2: forward cross (dω/dR_By · ∂²/∂pos_Ax∂ω)
  G3: reverse cross (dω/dR_Ax · ∂²/∂pos_By∂ω)
  G4: d²ω piece (d²ω/(dR_Ax dR_By) · ∂/∂ω)
  G5: ω×ω piece (dω/dR_Ax · dω/dR_By · ∂²/∂ω²)
"""

import numpy as np

from pyscf.solvent.gostshyp import fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A


def kernel(gost, dm):
    """Compute the full analytical GOSTSHYP Hessian contribution.

    Parameters
    ----------
    gost : GOSTSHYP object with kernel() already called
    dm : ndarray (nao, nao)

    Returns
    -------
    hess : ndarray (natm, natm, 3, 3)
    """
    mol = gost.mol
    natm = mol.natm
    P = gost.pressure_au
    e_g = gost.gtilde_expval
    F_g = gost.forces
    A_g = gost.areas

    # First derivatives of e_g and F_g
    dg_trace, dF_trace = _compute_scalar_traces(gost, dm)

    # Area derivatives (first and second)
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dA = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
        d2A = d2A * gost._occ_ratio_sq
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dA = dareas_raw.transpose(1, 2, 0)
        _, d2A = get_d2F_d2A(gost.surface_dict)

    # Second derivative traces
    d2e = _compute_d2e(gost, dm)
    d2F = _compute_d2F(gost, dm)

    # Assemble: E = P · Σ_g A·e/F → 10 terms from product rule
    inv_F = 1.0 / F_g
    inv_F2 = inv_F ** 2
    inv_F3 = inv_F ** 3

    hess = np.zeros((natm, natm, 3, 3))
    # H1: d²A · e/F
    hess += P * np.einsum('ABxyg,g,g->ABxy', d2A, e_g, inv_F, optimize=True)
    # H2: dA_Ax · de_By / F
    hess += P * np.einsum('Axg,Byg,g->ABxy', dA, dg_trace, inv_F,
                          optimize=True)
    # H3: -dA_Ax · e · dF_By / F²
    hess -= P * np.einsum('Axg,g,Byg,g->ABxy', dA, e_g, dF_trace, inv_F2,
                          optimize=True)
    # H4: dA_By · de_Ax / F
    hess += P * np.einsum('Byg,Axg,g->ABxy', dA, dg_trace, inv_F,
                          optimize=True)
    # H5: A/F · d²e
    hess += P * np.einsum('g,g,ABxyg->ABxy', A_g, inv_F, d2e, optimize=True)
    # H6: -A/F² · de_Ax · dF_By
    hess -= P * np.einsum('g,g,Axg,Byg->ABxy', A_g, inv_F2, dg_trace,
                          dF_trace, optimize=True)
    # H7: -dA_By · e · dF_Ax / F²
    hess -= P * np.einsum('Byg,g,Axg,g->ABxy', dA, e_g, dF_trace, inv_F2,
                          optimize=True)
    # H8: -A/F² · de_By · dF_Ax
    hess -= P * np.einsum('g,g,Byg,Axg->ABxy', A_g, inv_F2, dg_trace,
                          dF_trace, optimize=True)
    # H9: -A·e/F² · d²F
    hess -= P * np.einsum('g,g,g,ABxyg->ABxy', A_g, e_g, inv_F2, d2F,
                          optimize=True)
    # H10: 2·A·e/F³ · dF_Ax · dF_By
    hess += 2.0 * P * np.einsum('g,g,g,Axg,Byg->ABxy', A_g, e_g, inv_F3,
                                dF_trace, dF_trace, optimize=True)

    return 0.5 * (hess + hess.transpose(1, 0, 3, 2))


# ---------------------------------------------------------------------------
# First derivatives of scalar traces
# ---------------------------------------------------------------------------

def _compute_scalar_traces(gost, dm):
    """Compute de_g/dR_Ax and dF_g/dR_Ax for all atoms.

    Returns
    -------
    dg_trace : ndarray (natm, 3, ngrids)
    dF_trace : ndarray (natm, 3, ngrids)
    """
    mol = gost.mol
    nao_cart = mol.nao_nr(cart=True)
    natm = mol.natm
    ngrids = gost.n_gaussian
    aoslice = mol.aoslice_by_atom()
    widths = gost.widths
    areas = gost.areas
    normals = gost.surface_normals

    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized='sp')

    # Area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas.transpose(1, 2, 0) * gost._occ_ratio_sq
    else:
        _, dareas = get_dF_dA(gost.surface_dict)
        dareas = dareas.transpose(1, 2, 0)

    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)  # dω/dA

    # --- de_g/dR: s-type (Gtilde trace) ---
    gmol_s = fakemol_for_gaussian(gost.grid_coords, widths)
    supermol_s = mol + gmol_s
    nao = mol.nao_nr()
    slices_s = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_s.nbas)
    slices_sg = (mol.nbas, mol.nbas + gmol_s.nbas, 0, mol.nbas, 0, mol.nbas)

    dPQ_s = supermol_s.intor('int3c1e_ip1', shls_slice=slices_s)
    dG_s = supermol_s.intor('int3c1e_ip1', shls_slice=slices_sg)

    dg_trace = np.zeros((natm, 3, ngrids))
    dPQ_s_dm = np.einsum('xijg,ij->xig', dPQ_s, dm, optimize=True)
    dPQ_s_dm += np.einsum('xijg,ji->xig', dPQ_s, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dg_trace[A] -= np.sum(dPQ_s_dm[:, p0:p1, :], axis=1)
    dG_s_dm = np.einsum('xgij,ij->xg', dG_s, dm, optimize=True)
    for g in range(ngrids):
        dg_trace[gost.atom_idx[g], :, g] -= dG_s_dm[:, g]
    del dPQ_s, dG_s, dPQ_s_dm, dG_s_dm

    # Width contribution to de: dω/dR · Tr[D · ∂G/∂ω] = wgp·dA·(-diagd_dm)
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
    dg_trace -= np.einsum('axg,g->axg', dareas, wgrad_prefs * diagd_dm,
                          optimize=True)
    del overlap3d, diagd

    # --- dF_g/dR: p-type (Fhat trace) ---
    gmol_p = fakemol_for_gaussian(gost.grid_coords, widths, l=1,
                                  coeffs=2.0 * widths)
    supermol_p = mol + gmol_p
    slices_p = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)
    slices_pg = (mol.nbas, mol.nbas + gmol_p.nbas, 0, mol.nbas, 0, mol.nbas)

    dPQ_p_raw = supermol_p.intor('int3c1e_ip1', shls_slice=slices_p
                                 ).reshape(3, nao, nao, ngrids, 3)
    dPQ_p = np.einsum('xijgc,gc->xijg', dPQ_p_raw, normals, optimize=True)
    dG_p_raw = supermol_p.intor('int3c1e_ip1', shls_slice=slices_pg
                                ).reshape(3, ngrids, 3, nao, nao)
    dG_p = np.einsum('xgcij,gc->xgij', dG_p_raw, normals, optimize=True)
    del dPQ_p_raw, dG_p_raw

    dF_trace = np.zeros((natm, 3, ngrids))
    dPQ_p_dm = np.einsum('xijg,ij->xig', dPQ_p, dm, optimize=True)
    dPQ_p_dm += np.einsum('xijg,ji->xig', dPQ_p, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dF_trace[A] -= np.sum(dPQ_p_dm[:, p0:p1, :], axis=1)
    dG_p_dm = np.einsum('xgij,ij->xg', dG_p, dm, optimize=True)
    for g in range(ngrids):
        dF_trace[gost.atom_idx[g], :, g] -= dG_p_dm[:, g]
    del dPQ_p, dG_p, dPQ_p_dm, dG_p_dm

    # Width contribution to dF: dω/dR · Tr[D · ∂F/∂ω]
    # ∂F/∂ω = F/ω + f_contracted (l=3 Laplacian-traced, normal-contracted)
    gmol_f = fakemol_for_gaussian(gost.grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    overlap3f = supermol_f.intor('int3c1e', shls_slice=slices_f
                                 ).reshape(nao_cart, nao_cart, ngrids, 10)
    if not mol.cart:
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s,
                              optimize=True)
    fx = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    fy = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    fz = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
    f_contracted_dm = (
        np.einsum('ijg,ij->g', fx, dm, optimize=True) * normals[:, 0]
        + np.einsum('ijg,ij->g', fy, dm, optimize=True) * normals[:, 1]
        + np.einsum('ijg,ij->g', fz, dm, optimize=True) * normals[:, 2]
    )
    dFhat_domega_trace = gost.forces / widths + f_contracted_dm
    dF_trace += np.einsum('axg,g->axg', dareas,
                          wgrad_prefs * dFhat_domega_trace, optimize=True)
    del overlap3f, fx, fy, fz

    return dg_trace, dF_trace


# ---------------------------------------------------------------------------
# d²e_g/(dR_Ax dR_By) — second derivative of Gtilde trace
# ---------------------------------------------------------------------------

def _compute_d2e(gost, dm):
    """Compute d²e_g/(dR_Ax dR_By) analytically.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    mol = gost.mol
    nao = mol.nao_nr()
    nao_cart = mol.nao_nr(cart=True)
    natm = mol.natm
    ngrids = gost.n_gaussian
    aoslice = mol.aoslice_by_atom()
    widths = gost.widths
    areas = gost.areas
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)

    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized='sp')

    # Area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
        d2A = d2A * gost._occ_ratio_sq
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0)
        _, d2A = get_d2F_d2A(gost.surface_dict)

    d2e = np.zeros((natm, natm, 3, 3, ngrids))

    # === G1: Position × Position ===
    gmol_s = fakemol_for_gaussian(gost.grid_coords, widths)
    supermol = mol + gmol_s
    slices_s = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_s.nbas)

    # ipip1: ∂²/∂R_bra² (diagonal A=B)
    ipip1 = supermol.intor('int3c1e_ipip1', shls_slice=slices_s)
    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d2e[A, A] += np.sum(ipip1_dm[:, p0:p1, :], axis=1).reshape(3, 3, ngrids)
    del ipip1, ipip1_dm

    # ipvip1: ∂²/(∂R_bra ∂R_ket)
    ipvip1 = supermol.intor('int3c1e_ipvip1', shls_slice=slices_s)
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
            d2e[A, B] += trace_block.reshape(3, 3, ngrids)
    del ipvip1

    # ip1ip2: ∂²/(∂R_bra ∂R_aux)
    ip1ip2 = supermol.intor('int3c1e_ip1ip2', shls_slice=slices_s)
    ip1ip2_dm = np.einsum('xijg,ij->xig', ip1ip2, dm, optimize=True)
    ip1ip2_dm += np.einsum('xijg,ji->xig', ip1ip2, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib_A = np.sum(ip1ip2_dm[:, p0:p1, :], axis=1)
        for g in range(ngrids):
            B = gost.atom_idx[g]
            d2e[A, B, :, :, g] += contrib_A[:, g].reshape(3, 3)
    # Reverse: aux × bra/ket
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            trace_ig = np.sum(ip1ip2_dm[:, p0:p1, g], axis=1)
            d2e[A, B, :, :, g] += trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm

    # ipip2: ∂²/∂R_aux² (diagonal)
    ipip2 = supermol.intor('int3c1e_ipip2', shls_slice=slices_s)
    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2e[B, B, :, :, g] += ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    # === G2+G3: Position × Width cross terms ===
    gmol_d = fakemol_for_gaussian(gost.grid_coords, widths, l=2)
    supermol_d = mol + gmol_d
    supermol_d.cart = True
    slices_d = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_d.nbas)
    slices_dg = (mol.nbas, mol.nbas + gmol_d.nbas, 0, mol.nbas, 0, mol.nbas)

    # ip1 of d-type (bra derivative)
    ip1_d_bra = supermol_d.intor('int3c1e_ip1', shls_slice=slices_d
                                  ).reshape(3, nao_cart, nao_cart, ngrids, 6)
    ip1_diagd_bra = (ip1_d_bra[:, :, :, :, 0] + ip1_d_bra[:, :, :, :, 3]
                     + ip1_d_bra[:, :, :, :, 5])
    del ip1_d_bra
    if not mol.cart:
        ip1_diagd_bra = np.einsum('mi,xijg,jn->xmng', c2s.T,
                                  ip1_diagd_bra, c2s, optimize=True)
    ip1_diagd_dm_bra = np.einsum('xijg,ij->xig', ip1_diagd_bra, dm,
                                 optimize=True)
    ip1_diagd_dm_bra += np.einsum('xijg,ji->xig', ip1_diagd_bra, dm,
                                  optimize=True)
    del ip1_diagd_bra

    d_diagd_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_diagd_dm_pos[A] -= np.sum(ip1_diagd_dm_bra[:, p0:p1, :], axis=1)
    del ip1_diagd_dm_bra

    # ip1 of d-type (aux derivative)
    ip1_d_aux = supermol_d.intor('int3c1e_ip1', shls_slice=slices_dg
                                  ).reshape(3, ngrids, 6, nao_cart, nao_cart)
    ip1_diagd_aux = (ip1_d_aux[:, :, 0, :, :] + ip1_d_aux[:, :, 3, :, :]
                     + ip1_d_aux[:, :, 5, :, :])
    del ip1_d_aux
    if not mol.cart:
        ip1_diagd_aux = np.einsum('mi,xgij,jn->xgmn', c2s.T,
                                  ip1_diagd_aux, c2s, optimize=True)
    ip1_diagd_dm_aux = np.einsum('xgij,ij->xg', ip1_diagd_aux, dm,
                                 optimize=True)
    del ip1_diagd_aux
    for g in range(ngrids):
        d_diagd_dm_pos[gost.atom_idx[g], :, g] -= ip1_diagd_dm_aux[:, g]
    del ip1_diagd_dm_aux

    # G2 + G3: ∂²e/(∂pos ∂ω) = -∂diagd/∂pos → cross = -wgp·dA·d_diagd_dm_pos
    d2e += np.einsum('g,Byg,Axg->ABxyg', -wgrad_prefs, dareas,
                     d_diagd_dm_pos, optimize=True)
    d2e += np.einsum('g,Axg,Byg->ABxyg', -wgrad_prefs, dareas,
                     d_diagd_dm_pos, optimize=True)

    # === G4+G5: Width × Width terms ===
    # G4: d²ω · ∂e/∂ω = d²ω · (-diagd_dm)
    gmol_d2 = fakemol_for_gaussian(gost.grid_coords, widths, l=2)
    supermol_d2 = mol + gmol_d2
    supermol_d2.cart = True
    slices_d2 = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_d2.nbas)
    overlap3d = supermol_d2.intor('int3c1e', shls_slice=slices_d2
                                  ).reshape(nao_cart, nao_cart, ngrids, 6)
    if not mol.cart:
        overlap3d = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s,
                              optimize=True)
    diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
    diagd_dm = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
    del overlap3d, diagd

    # G4a: (-2·wgp/A · dA_Ax · dA_By) · (-diagd_dm)
    d2e += np.einsum('g,Axg,Byg,g->ABxyg',
                     2.0 * wgrad_prefs / areas, dareas, dareas, diagd_dm,
                     optimize=True)
    # G4b: (wgp · d²A) · (-diagd_dm)
    d2e -= np.einsum('g,ABxyg,g->ABxyg', wgrad_prefs, d2A, diagd_dm,
                     optimize=True)

    # G5: dω_Ax · dω_By · ∂²e/∂ω² (l=4 g-type)
    gmol_g4 = fakemol_for_gaussian(gost.grid_coords, widths, l=4)
    supermol_g4 = mol + gmol_g4
    supermol_g4.cart = True
    slices_g4 = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_g4.nbas)
    overlap3g = supermol_g4.intor('int3c1e', shls_slice=slices_g4
                                  ).reshape(nao_cart, nao_cart, ngrids, 15)
    if not mol.cart:
        overlap3g = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3g, c2s,
                              optimize=True)
    # (r²)² = x⁴+y⁴+z⁴+2x²y²+2x²z²+2y²z²
    gtype_trace = (overlap3g[:, :, :, 0] + overlap3g[:, :, :, 10]
                   + overlap3g[:, :, :, 14]
                   + 2.0 * overlap3g[:, :, :, 3]
                   + 2.0 * overlap3g[:, :, :, 5]
                   + 2.0 * overlap3g[:, :, :, 12])
    gtype_dm = np.einsum('ijg,ij->g', gtype_trace, dm, optimize=True)
    del overlap3g, gtype_trace

    d2e += np.einsum('g,Axg,g,Byg,g->ABxyg',
                     wgrad_prefs, dareas, wgrad_prefs, dareas, gtype_dm,
                     optimize=True)

    return d2e


# ---------------------------------------------------------------------------
# d²F_g/(dR_Ax dR_By) — second derivative of Fhat (force) trace
# ---------------------------------------------------------------------------

def _compute_d2F(gost, dm, force_thresh=1e-9):
    """Compute d²F_g/(dR_Ax dR_By) analytically.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    mol = gost.mol
    nao = mol.nao_nr()
    nao_cart = mol.nao_nr(cart=True)
    natm = mol.natm
    ngrids = gost.n_gaussian
    aoslice = mol.aoslice_by_atom()
    widths = gost.widths
    areas = gost.areas
    normals = gost.surface_normals
    forces = gost.forces
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)

    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized='sp')

    # Area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
        d2A = d2A * gost._occ_ratio_sq
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0)
        _, d2A = get_d2F_d2A(gost.surface_dict)

    # Conditioning guard for width-response terms
    stable = forces > force_thresh

    d2F = np.zeros((natm, natm, 3, 3, ngrids))

    # === G1: Position × Position (p-type + normals) ===
    gmol_p = fakemol_for_gaussian(gost.grid_coords, widths, l=1,
                                  coeffs=2.0 * widths)
    supermol = mol + gmol_p
    slices_p = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p.nbas)

    # ipip1
    ipip1_raw = supermol.intor('int3c1e_ipip1', shls_slice=slices_p
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip1 = np.einsum('xijgc,gc->xijg', ipip1_raw, normals, optimize=True)
    del ipip1_raw
    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d2F[A, A] += np.sum(ipip1_dm[:, p0:p1, :], axis=1).reshape(3, 3, ngrids)
    del ipip1, ipip1_dm

    # ipvip1
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
            d2F[A, B] += trace_block.reshape(3, 3, ngrids)
    del ipvip1

    # ip1ip2
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
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            trace_ig = np.sum(ip1ip2_dm[:, p0:p1, g], axis=1)
            d2F[A, B, :, :, g] += trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm

    # ipip2
    ipip2_raw = supermol.intor('int3c1e_ipip2', shls_slice=slices_p
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip2 = np.einsum('xijgc,gc->xijg', ipip2_raw, normals, optimize=True)
    del ipip2_raw
    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2F[B, B, :, :, g] += ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    # === G2+G3: Position × Width cross terms ===
    # ∂²F/(∂pos_Ax ∂ω) = (1/ω)·dF_pos[A,x] + d_f_dm_pos[A,x]
    slices_pg = (mol.nbas, mol.nbas + gmol_p.nbas,
                 0, mol.nbas, 0, mol.nbas)

    # dF_pos: position-only part of dF/dR (p-type ip1)
    ip1_p_bra_raw = supermol.intor('int3c1e_ip1', shls_slice=slices_p
                                   ).reshape(3, nao, nao, ngrids, 3)
    ip1_p_bra = np.einsum('xijgc,gc->xijg', ip1_p_bra_raw, normals,
                          optimize=True)
    del ip1_p_bra_raw
    ip1_p_bra_dm = np.einsum('xijg,ij->xig', ip1_p_bra, dm, optimize=True)
    ip1_p_bra_dm += np.einsum('xijg,ji->xig', ip1_p_bra, dm, optimize=True)
    del ip1_p_bra

    ip1_p_aux_raw = supermol.intor('int3c1e_ip1', shls_slice=slices_pg
                                   ).reshape(3, ngrids, 3, nao, nao)
    ip1_p_aux = np.einsum('xgcij,gc->xgij', ip1_p_aux_raw, normals,
                          optimize=True)
    del ip1_p_aux_raw
    ip1_p_aux_dm = np.einsum('xgij,ij->xg', ip1_p_aux, dm, optimize=True)
    del ip1_p_aux

    dF_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        dF_pos[A] -= np.sum(ip1_p_bra_dm[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        dF_pos[gost.atom_idx[g], :, g] -= ip1_p_aux_dm[:, g]
    del ip1_p_bra_dm, ip1_p_aux_dm

    # d_f_dm_pos: position derivative of f_contracted_dm (l=3)
    gmol_f = fakemol_for_gaussian(gost.grid_coords, widths, l=3,
                                  coeffs=-2.0 * widths)
    supermol_f = mol + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    slices_fg = (mol.nbas, mol.nbas + gmol_f.nbas, 0, mol.nbas, 0, mol.nbas)

    ip1_f_bra = supermol_f.intor('int3c1e_ip1', shls_slice=slices_f
                                 ).reshape(3, nao_cart, nao_cart, ngrids, 10)
    ip1_fx = (ip1_f_bra[:, :, :, :, 0] + ip1_f_bra[:, :, :, :, 3]
              + ip1_f_bra[:, :, :, :, 5])
    ip1_fy = (ip1_f_bra[:, :, :, :, 1] + ip1_f_bra[:, :, :, :, 6]
              + ip1_f_bra[:, :, :, :, 8])
    ip1_fz = (ip1_f_bra[:, :, :, :, 2] + ip1_f_bra[:, :, :, :, 7]
              + ip1_f_bra[:, :, :, :, 9])
    del ip1_f_bra
    ip1_f_contracted_bra = (ip1_fx * normals[:, 0] + ip1_fy * normals[:, 1]
                            + ip1_fz * normals[:, 2])
    del ip1_fx, ip1_fy, ip1_fz
    if not mol.cart:
        ip1_f_contracted_bra = np.einsum('mi,xijg,jn->xmng', c2s.T,
                                         ip1_f_contracted_bra, c2s,
                                         optimize=True)
    ip1_f_dm_bra = np.einsum('xijg,ij->xig', ip1_f_contracted_bra, dm,
                             optimize=True)
    ip1_f_dm_bra += np.einsum('xijg,ji->xig', ip1_f_contracted_bra, dm,
                              optimize=True)
    del ip1_f_contracted_bra

    ip1_f_aux = supermol_f.intor('int3c1e_ip1', shls_slice=slices_fg
                                 ).reshape(3, ngrids, 10, nao_cart, nao_cart)
    ip1_fx_aux = (ip1_f_aux[:, :, 0, :, :] + ip1_f_aux[:, :, 3, :, :]
                  + ip1_f_aux[:, :, 5, :, :])
    ip1_fy_aux = (ip1_f_aux[:, :, 1, :, :] + ip1_f_aux[:, :, 6, :, :]
                  + ip1_f_aux[:, :, 8, :, :])
    ip1_fz_aux = (ip1_f_aux[:, :, 2, :, :] + ip1_f_aux[:, :, 7, :, :]
                  + ip1_f_aux[:, :, 9, :, :])
    del ip1_f_aux
    ip1_f_contracted_aux = (ip1_fx_aux * normals[:, 0][:, None, None]
                            + ip1_fy_aux * normals[:, 1][:, None, None]
                            + ip1_fz_aux * normals[:, 2][:, None, None])
    del ip1_fx_aux, ip1_fy_aux, ip1_fz_aux
    if not mol.cart:
        ip1_f_contracted_aux = np.einsum('mi,xgij,jn->xgmn', c2s.T,
                                         ip1_f_contracted_aux, c2s,
                                         optimize=True)
    ip1_f_dm_aux = np.einsum('xgij,ij->xg', ip1_f_contracted_aux, dm,
                             optimize=True)
    del ip1_f_contracted_aux

    d_f_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_f_dm_pos[A] -= np.sum(ip1_f_dm_bra[:, p0:p1, :], axis=1)
    for g in range(ngrids):
        d_f_dm_pos[gost.atom_idx[g], :, g] -= ip1_f_dm_aux[:, g]
    del ip1_f_dm_bra, ip1_f_dm_aux

    # Combined cross-term scalar
    d_dFhat_domega_dm_pos = dF_pos / widths + d_f_dm_pos
    d_dFhat_domega_dm_pos[:, :, ~stable] = 0.0

    # G2 + G3
    d2F += np.einsum('g,Byg,Axg->ABxyg', wgrad_prefs, dareas,
                     d_dFhat_domega_dm_pos, optimize=True)
    d2F += np.einsum('g,Axg,Byg->ABxyg', wgrad_prefs, dareas,
                     d_dFhat_domega_dm_pos, optimize=True)

    # === G4: d²ω · ∂F/∂ω ===
    # f_contracted_dm from l=3 overlap
    overlap3f = supermol_f.intor('int3c1e', shls_slice=slices_f
                                 ).reshape(nao_cart, nao_cart, ngrids, 10)
    if not mol.cart:
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s,
                              optimize=True)
    fx = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    fy = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    fz = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
    f_contracted_dm = (
        np.einsum('ijg,ij->g', fx, dm, optimize=True) * normals[:, 0]
        + np.einsum('ijg,ij->g', fy, dm, optimize=True) * normals[:, 1]
        + np.einsum('ijg,ij->g', fz, dm, optimize=True) * normals[:, 2]
    )
    del overlap3f, fx, fy, fz

    dFhat_domega_trace = forces / widths + f_contracted_dm
    dFhat_domega_trace[~stable] = 0.0

    # G4a: (-2·wgp/A · dA_Ax · dA_By) · dFhat_domega_trace
    d2F += np.einsum('g,Axg,Byg,g->ABxyg',
                     -2.0 * wgrad_prefs / areas, dareas, dareas,
                     dFhat_domega_trace, optimize=True)
    # G4b: wgp · d²A · dFhat_domega_trace
    d2F += np.einsum('g,ABxyg,g->ABxyg', wgrad_prefs, d2A,
                     dFhat_domega_trace, optimize=True)

    # === G5: dω_Ax · dω_By · ∂²F/∂ω² ===
    # ∂²F/∂ω² = (2/ω)·f_contracted_dm + h5_dm (l=5)
    gmol_h5 = fakemol_for_gaussian(gost.grid_coords, widths, l=5,
                                   coeffs=2.0 * widths)
    supermol_h5 = mol + gmol_h5
    supermol_h5.cart = True
    slices_h5 = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_h5.nbas)
    overlap3h = supermol_h5.intor('int3c1e', shls_slice=slices_h5
                                  ).reshape(nao_cart, nao_cart, ngrids, 21)
    if not mol.cart:
        overlap3h = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3h, c2s,
                              optimize=True)
    # r_c · r⁴ contracted with normals
    h5x = (overlap3h[:, :, :, 0] + 2*overlap3h[:, :, :, 3]
            + 2*overlap3h[:, :, :, 5] + overlap3h[:, :, :, 10]
            + 2*overlap3h[:, :, :, 12] + overlap3h[:, :, :, 14])
    h5y = (overlap3h[:, :, :, 1] + 2*overlap3h[:, :, :, 6]
            + 2*overlap3h[:, :, :, 8] + overlap3h[:, :, :, 15]
            + 2*overlap3h[:, :, :, 17] + overlap3h[:, :, :, 19])
    h5z = (overlap3h[:, :, :, 2] + 2*overlap3h[:, :, :, 7]
            + 2*overlap3h[:, :, :, 9] + overlap3h[:, :, :, 16]
            + 2*overlap3h[:, :, :, 18] + overlap3h[:, :, :, 20])
    h5_dm = (
        np.einsum('ijg,ij->g', h5x, dm, optimize=True) * normals[:, 0]
        + np.einsum('ijg,ij->g', h5y, dm, optimize=True) * normals[:, 1]
        + np.einsum('ijg,ij->g', h5z, dm, optimize=True) * normals[:, 2]
    )
    del overlap3h, h5x, h5y, h5z

    d2Fhat_domega2_trace = 2.0 * f_contracted_dm / widths + h5_dm
    d2Fhat_domega2_trace[~stable] = 0.0

    d2F += np.einsum('g,Axg,g,Byg,g->ABxyg',
                     wgrad_prefs, dareas, wgrad_prefs, dareas,
                     d2Fhat_domega2_trace, optimize=True)

    return d2F
