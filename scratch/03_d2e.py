#!/usr/bin/env python
"""
Test 3: Validate d²e_g/(dR_Ax dR_By) — second derivative of Gtilde trace.

This is the hardest new computation for the Hessian (term H5).
d²e_g/(dR_Ax dR_By) = Tr[D · d²Gtilde_g/(dR_Ax dR_By)]

Sub-terms:
  Position × Position: ipip1, ipvip1, ip1ip2, ipip2
  Position × Width: ip1 of d-type integrals
  Width × Width: g-type (l=4) integrals
"""

import numpy as np
from pyscf import gto
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A

from common import (make_h2_system, make_h2o_system, compute_scalar_traces,
                    get_gost_options)


def compute_d2e_fd(gost, dm, mol, step=1e-4):
    """Compute d²e_g/(dR_Ax dR_By) by finite-differencing de_g/dR.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    natm = mol.natm
    ngrids = gost.n_gaussian
    coords0 = mol.atom_coords().copy()
    opts = get_gost_options(gost)

    d2e_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy()
            coords_p[B, y] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=opts)
            gost_p.kernel(dm)
            dg_p, _ = compute_scalar_traces(gost_p, dm)

            coords_m = coords0.copy()
            coords_m[B, y] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=opts)
            gost_m.kernel(dm)
            dg_m, _ = compute_scalar_traces(gost_m, dm)

            d2e_fd[:, B, :, y, :] = (dg_p - dg_m) / (2.0 * step)

    return d2e_fd


def compute_d2e_analytical(gost, dm, mol):
    """Compute d²e_g/(dR_Ax dR_By) analytically — ALL terms.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    mol_obj = gost.mol
    nao = mol_obj.nao_nr()
    nao_cart = mol_obj.nao_nr(cart=True)
    natm = mol_obj.natm
    ngrids = gost.n_gaussian
    aoslice = mol_obj.aoslice_by_atom()

    widths = gost.widths
    areas = gost.areas
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)  # dω/dA

    if not mol_obj.cart:
        c2s = mol_obj.cart2sph_coeff(normalized='sp')

    # Area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
        d2A = d2A * gost._occ_ratio_sq
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0)  # (natm, 3, ngrids)
        _, d2A = get_d2F_d2A(gost.surface_dict)

    d2e = np.zeros((natm, natm, 3, 3, ngrids))

    # =====================================================================
    # PART 1: Position × Position (second-derivative integrals)
    # =====================================================================
    # Sign convention: PySCF's int3c1e_ipXX integrals compute derivatives
    # w.r.t. ELECTRON coordinates (∂²/∂r²). For nuclear coordinates:
    #   ∂²I/∂R² = (-1)(-1) ∂²I/∂r² = +∂²I/∂r²
    # Two sign flips cancel, so we use += (NOT -= as in the gradient where
    # there is only one derivative and ip1 = -∂I/∂R).
    # =====================================================================
    gmol_s = fakemol_for_gaussian(gost.grid_coords, widths)
    supermol = mol_obj + gmol_s
    slices_s = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_s.nbas)
    slices_sg = (mol_obj.nbas, mol_obj.nbas + gmol_s.nbas,
                 0, mol_obj.nbas, 0, mol_obj.nbas)

    # ipip1: ∂²/∂R_bra² (A=B, both on bra atom)
    ipip1 = supermol.intor('int3c1e_ipip1', shls_slice=slices_s)
    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib = np.sum(ipip1_dm[:, p0:p1, :], axis=1)
        d2e[A, A, :, :, :] += contrib.reshape(3, 3, ngrids)
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
            d2e[A, B, :, :, :] += trace_block.reshape(3, 3, ngrids)
    del ipvip1

    # ip1ip2: ∂²/(∂R_bra ∂R_aux) — bra on A, aux on B
    ip1ip2 = supermol.intor('int3c1e_ip1ip2', shls_slice=slices_s)
    ip1ip2_dm = np.einsum('xijg,ij->xig', ip1ip2, dm, optimize=True)
    ip1ip2_dm += np.einsum('xijg,ji->xig', ip1ip2, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib_A = np.sum(ip1ip2_dm[:, p0:p1, :], axis=1)  # (9, ngrids)
        for g in range(ngrids):
            B = gost.atom_idx[g]
            d2e[A, B, :, :, g] += contrib_A[:, g].reshape(3, 3)

    # aux × bra/ket: ∂²/(∂R_aux ∂R_bra) — transpose of ip1ip2
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            trace_ig = np.sum(ip1ip2_dm[:, p0:p1, g], axis=1)
            d2e[A, B, :, :, g] += trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm

    # ipip2: ∂²/∂R_aux² (both on aux, same atom)
    ipip2 = supermol.intor('int3c1e_ipip2', shls_slice=slices_s)
    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2e[B, B, :, :, g] += ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    # =====================================================================
    # PART 2: Position × Width cross terms
    # =====================================================================
    # When the first derivative is "position" (ip1) and the second acts
    # through ω: ∂(ip1)/∂ω · dω/dR_By
    #
    # ∂(ip1_x of s-type)/∂ω = -(ip1_x of diagd)
    # where diagd = Laplacian trace of d-type integral
    #
    # Contribution: dω/dR_By · Tr[D · ∂²Gtilde/(∂pos_Ax ∂ω)]
    # = -wgp · dA[B,y,g] · (ip1_of_diagd traced with D, restricted to atom A)

    gmol_d = fakemol_for_gaussian(gost.grid_coords, widths, l=2)
    supermol_d = mol_obj + gmol_d
    supermol_d.cart = True
    slices_d = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_d.nbas)
    slices_dg = (mol_obj.nbas, mol_obj.nbas + gmol_d.nbas,
                 0, mol_obj.nbas, 0, mol_obj.nbas)

    # ip1 of d-type (bra derivative): shape (3, nao_cart, nao_cart, ngrids*6)
    ip1_d_bra = supermol_d.intor('int3c1e_ip1', shls_slice=slices_d
                                  ).reshape(3, nao_cart, nao_cart, ngrids, 6)
    # Laplacian trace: xx(0) + yy(3) + zz(5)
    ip1_diagd_bra = ip1_d_bra[:, :, :, :, 0] + ip1_d_bra[:, :, :, :, 3] + ip1_d_bra[:, :, :, :, 5]
    del ip1_d_bra
    # Transform to spherical AOs if needed
    if not mol_obj.cart:
        ip1_diagd_bra = np.einsum('mi,xijg,jn->xmng', c2s.T, ip1_diagd_bra, c2s,
                                  optimize=True)
    # Shape now: (3, nao, nao, ngrids)

    # Trace with D (bra+ket): for each AO i, sum over j
    ip1_diagd_dm_bra = np.einsum('xijg,ij->xig', ip1_diagd_bra, dm, optimize=True)
    ip1_diagd_dm_bra += np.einsum('xijg,ji->xig', ip1_diagd_bra, dm, optimize=True)
    del ip1_diagd_bra

    # Build d(diagd_dm)/dR_Ax (position part) — same structure as de/dR
    # This is needed for Part 3 as well
    d_diagd_dm_pos = np.zeros((natm, 3, ngrids))
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        d_diagd_dm_pos[A, :, :] -= np.sum(ip1_diagd_dm_bra[:, p0:p1, :], axis=1)

    # ip1 of d-type (aux derivative): shape (3, ngrids*6, nao_cart, nao_cart)
    ip1_d_aux = supermol_d.intor('int3c1e_ip1', shls_slice=slices_dg
                                  ).reshape(3, ngrids, 6, nao_cart, nao_cart)
    # Laplacian trace
    ip1_diagd_aux = ip1_d_aux[:, :, 0, :, :] + ip1_d_aux[:, :, 3, :, :] + ip1_d_aux[:, :, 5, :, :]
    del ip1_d_aux
    if not mol_obj.cart:
        ip1_diagd_aux = np.einsum('mi,xgij,jn->xgmn', c2s.T, ip1_diagd_aux, c2s,
                                  optimize=True)
    # Shape: (3, ngrids, nao, nao)
    ip1_diagd_dm_aux = np.einsum('xgij,ij->xg', ip1_diagd_aux, dm, optimize=True)
    del ip1_diagd_aux

    # Add aux contribution to d_diagd_dm_pos
    for g in range(ngrids):
        atom_g = gost.atom_idx[g]
        d_diagd_dm_pos[atom_g, :, g] -= ip1_diagd_dm_aux[:, g]

    # --- Term (i): position × width ---
    # d²e[A,B,x,y,g] += dω/dR_By · Tr[D · ∂²Gtilde/(∂pos_Ax ∂ω)]
    #
    # For bra/ket on A:
    #   = -wgp[g] · dA[B,y,g] · (-Σ_{i∈A} (D+D^T) · ip1_diagd[x,i,:,g])
    #   = wgp · dA[B,y,g] · Σ_{i∈A} ip1_diagd_dm_bra[x,i,g]
    #
    # But ip1_diagd_dm_bra contributes to d_diagd_dm_pos with a minus sign,
    # so the bra/ket part of d_diagd_dm_pos[A,x,g] = -Σ_{i∈A} ip1_diagd_dm_bra[x,i,g] - aux
    # Let's use the combined d_diagd_dm_pos directly.
    #
    # The position×width cross term for d²e is:
    # dω/dR_By · Tr[D · ∂²G/(∂pos_Ax ∂ω)] = -wgp·dA[B,y,g] · d_diagd_dm_pos_from_A[A,x,g]
    #
    # Wait, need to be careful. ∂²G/(∂pos_Ax ∂ω) is the cross partial.
    # ∂G/∂pos_Ax gives ip1 terms. ∂/∂ω of that gives -(ip1 of diagd).
    # So ∂²G/(∂pos_Ax ∂ω) = -(ip1_diagd restricted to atom A)
    # Tr[D · ∂²G/(∂pos_Ax ∂ω)] = -(ip1_diagd_dm restricted to A)
    #                             = d_diagd_dm_pos[A,x,g] (which already has the negative)
    #
    # Hmm no. d_diagd_dm_pos[A,x,g] = Tr[D · ∂(diagd)/∂R_Ax] (position deriv of diagd)
    # That's different from Tr[D · ∂(ip1_braA)/∂ω]...
    #
    # Let me be very explicit:
    # Tr[D · ∂²Gtilde/(∂R_bra_Ax ∂ω)] for bra on A:
    #   = Σ_{i∈A,j} (D_ij + D_ji) · ∂²<i|G|j>/(∂R_i_x ∂ω)
    #   = Σ_{i∈A,j} (D_ij + D_ji) · (-(ip1_x of diagd)[i,j,g])
    #   = -Σ_{i∈A} ip1_diagd_dm_bra[x,i,g]
    #
    # Tr[D · ∂²Gtilde/(∂R_aux_Ax ∂ω)] for aux on A (g on A):
    #   = Σ_{i,j} D_ij · ∂²<i|G|j>/(∂R_g_x ∂ω)
    #   = Σ_{i,j} D_ij · (-(ip1_diagd_aux_x)[g,i,j])
    #   = -ip1_diagd_dm_aux[x,g]
    #
    # Combined: Tr[D · ∂²Gtilde/(∂pos_Ax ∂ω)] = d_diagd_dm_pos[A,x,g]
    # Because d_diagd_dm_pos already has the negative signs baked in!
    # (d_diagd_dm_pos = -bra_sum - aux_sum, which matches the structure)
    #
    # Wait no — d_diagd_dm_pos is ∂(diagd_dm)/∂R_Ax (position part), which is:
    #   Tr[D · ∂(diagd_g)/∂R_Ax]
    # And ∂²Gtilde/(∂pos_Ax ∂ω) = ∂/∂ω(∂Gtilde/∂pos_Ax) = -(∂diagd/∂pos_Ax)
    # (because ∂Gtilde/∂ω = -diagd, so ∂²Gtilde/(∂pos ∂ω) = -∂diagd/∂pos)
    #
    # So Tr[D · ∂²Gtilde/(∂pos_Ax ∂ω)] = -Tr[D · ∂diagd_g/∂R_Ax] = -d_diagd_dm_pos[A,x,g]
    #
    # Therefore term (i): dω/dR_By · Tr[D · ∂²Gtilde/(∂pos_Ax ∂ω)]
    #   = wgp·dA[B,y,g] · (-d_diagd_dm_pos[A,x,g])
    #   = -wgp·dA[B,y,g] · d_diagd_dm_pos[A,x,g]

    # Term (i): position × width
    d2e += np.einsum('g,Byg,Axg->ABxyg', -wgrad_prefs, dareas, d_diagd_dm_pos,
                     optimize=True)

    # Also need the REVERSE: width × position
    # dω/dR_Ax · Tr[D · ∂²Gtilde/(∂ω ∂pos_By)]
    # By symmetry of mixed partials: ∂²Gtilde/(∂ω ∂pos_By) = ∂²Gtilde/(∂pos_By ∂ω)
    # So this = wgp·dA[A,x,g] · (-d_diagd_dm_pos[B,y,g])
    d2e += np.einsum('g,Axg,Byg->ABxyg', -wgrad_prefs, dareas, d_diagd_dm_pos,
                     optimize=True)

    # =====================================================================
    # PART 3: Width × Width terms
    # =====================================================================
    # From differentiating the width contribution to de/dR_Ax:
    #   de/dR_Ax (width) = dω/dR_Ax · (-diagd_dm_g)
    #                     = wgp · dA[A,x,g] · (-diagd_dm_g)
    #
    # d/dR_By [wgp · dA[A,x,g] · (-diagd_dm_g)] =
    #   (a) d(wgp)/dR_By · dA[A,x,g] · (-diagd_dm)
    #   (b) wgp · d²A[A,B,x,y,g] · (-diagd_dm)
    #   (c) wgp · dA[A,x,g] · d(-diagd_dm)/dR_By
    #
    # But wait — terms (a) and (b) are part of d²ω/(dR_Ax dR_By) · (-diagd_dm)
    # And term (c) is dω/dR_Ax · d(-diagd_dm)/dR_By
    #
    # Actually, these are ALREADY partially captured in terms (i) above!
    # Term (i) handles: dω/dR_By · ∂(ip1_braA and aux_A contributions)/∂ω
    # But the width term's own derivative is SEPARATE from term (i).
    #
    # Let me recount. The full de/dR_Ax = pos_part + width_part:
    #   pos_part = Tr[D · ∂Gtilde/∂pos_Ax]    (from ip1)
    #   width_part = dω/dR_Ax · Tr[D · ∂Gtilde/∂ω] = -dω/dR_Ax · diagd_dm
    #
    # For d²e/(dR_Ax dR_By) = d(pos_part)/dR_By + d(width_part)/dR_By
    #
    # d(pos_part)/dR_By:
    #   = position×position (Part 1, done)
    #   + position×width: ∂(pos_part)/∂ω · dω/dR_By = Term (i) above
    #
    # d(width_part)/dR_By = d[-dω/dR_Ax · diagd_dm]/dR_By:
    #   = -d²ω/(dR_Ax dR_By) · diagd_dm         ...(ii)
    #   + (-dω/dR_Ax) · d(diagd_dm)/dR_By        ...(iii)
    #
    # where d(diagd_dm)/dR_By has position and width parts:
    #   d(diagd_dm)/dR_By = d_diagd_dm_pos[B,y,g]  (position, computed above)
    #                     + dω/dR_By · ∂(diagd_dm)/∂ω   (width)
    #
    # ∂(diagd_dm)/∂ω = Tr[D · ∂(diagd)/∂ω] = Tr[D · (-gtype_trace)]
    # where gtype_trace = (r-Rg)⁴·exp(-ω|r-Rg|²) Laplacian-of-Laplacian
    #
    # So term (iii) splits into:
    #   (iii_a): -dω/dR_Ax · d_diagd_dm_pos[B,y,g]     (width × position)
    #   (iii_b): -dω/dR_Ax · dω/dR_By · (-gtype_dm)    (width × width)
    #
    # But (iii_a) = -wgp·dA[A,x,g] · d_diagd_dm_pos[B,y,g]
    # This is EXACTLY the "reverse" term I already added above in term (i)!
    # So I've double-counted!
    #
    # Wait no. Term (i) was:
    #   dω/dR_By · Tr[D · ∂²G/(∂pos_Ax ∂ω)] — this handles d(pos_part)/dR_By through ω
    # And the reverse was:
    #   dω/dR_Ax · Tr[D · ∂²G/(∂pos_By ∂ω)] — handles d(pos_part)/dR_Ax... no wait.
    #
    # I'm confusing myself. Let me restart the counting very carefully.
    #
    # de_g/dR_Ax = Tr[D · dGtilde_g/dR_Ax]   (total derivative)
    #
    # dGtilde_g/dR_Ax = ∂Gtilde/∂pos_Ax + dω/dR_Ax · ∂Gtilde/∂ω
    #
    # d²e/(dR_Ax dR_By) = Tr[D · d²Gtilde/(dR_Ax dR_By)]
    #
    # d²Gtilde/(dR_Ax dR_By) = d/dR_By [∂Gtilde/∂pos_Ax + dω/dR_Ax · ∂Gtilde/∂ω]
    #
    # = ∂²Gtilde/(∂pos_Ax ∂pos_By)           [Part 1: done]
    #   + dω/dR_By · ∂²Gtilde/(∂pos_Ax ∂ω)  [Part 2a: term (i)]
    #   + d²ω/(dR_Ax dR_By) · ∂Gtilde/∂ω    [Part 3a]
    #   + dω/dR_Ax · ∂²Gtilde/(∂pos_By ∂ω)  [Part 2b: should NOT be here!]
    #   + dω/dR_Ax · dω/dR_By · ∂²Gtilde/∂ω²  [Part 3b]
    #
    # Wait, the last two come from d/dR_By [dω/dR_Ax · ∂Gtilde/∂ω]:
    # = d(dω/dR_Ax)/dR_By · ∂Gtilde/∂ω + dω/dR_Ax · d(∂Gtilde/∂ω)/dR_By
    # = d²ω/(dR_Ax dR_By) · ∂Gtilde/∂ω
    #   + dω/dR_Ax · [∂²Gtilde/(∂pos_By ∂ω) + dω/dR_By · ∂²Gtilde/∂ω²]
    #
    # So the full decomposition is:
    # d²Gtilde/(dR_Ax dR_By) =
    #   (1) ∂²Gtilde/(∂pos_Ax ∂pos_By)             [Part 1]
    #   (2) dω/dR_By · ∂²Gtilde/(∂pos_Ax ∂ω)      [cross: first pos, second width]
    #   (3) dω/dR_Ax · ∂²Gtilde/(∂pos_By ∂ω)      [cross: first width, second pos]
    #   (4) d²ω/(dR_Ax dR_By) · ∂Gtilde/∂ω        [pure width: d²ω piece]
    #   (5) dω/dR_Ax · dω/dR_By · ∂²Gtilde/∂ω²   [pure width: ∂²G/∂ω² piece]
    #
    # Terms (2) and (3) are the "position × width" cross terms.
    # Terms (4) and (5) are the "width × width" terms.
    #
    # I already computed (2) and (3) above — good, no double counting.
    # (They're symmetric: (2) uses d_diagd_dm_pos[A,x] with dA[B,y],
    #  and (3) uses d_diagd_dm_pos[B,y] with dA[A,x])

    # --- Compute diagd_dm for terms (4) ---
    # diagd_dm = Tr[D · diagd_g] (already available conceptually)
    gmol_d2 = fakemol_for_gaussian(gost.grid_coords, widths, l=2)
    supermol_d2 = mol_obj + gmol_d2
    supermol_d2.cart = True
    slices_d2 = (0, mol_obj.nbas, 0, mol_obj.nbas,
                 mol_obj.nbas, mol_obj.nbas + gmol_d2.nbas)
    overlap3d = supermol_d2.intor('int3c1e', shls_slice=slices_d2
                                  ).reshape(nao_cart, nao_cart, ngrids, 6)
    if not mol_obj.cart:
        overlap3d = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s, optimize=True)
    diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
    diagd_dm = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
    del overlap3d, diagd

    # Term (4): d²ω/(dR_Ax dR_By) · Tr[D · ∂Gtilde/∂ω]
    # ∂Gtilde/∂ω = -diagd, so Tr[D · ∂Gtilde/∂ω] = -diagd_dm
    # d²ω/(dR_Ax dR_By) = d(wgp·dA[A,x,g])/dR_By
    #   = dwgp/dR_By · dA[A,x,g] + wgp · d²A[A,B,x,y,g]
    # where dwgp/dR_By = d(-πln2/A²)/dR_By = 2πln2/A³ · dA[B,y,g]
    #   = -2·wgp/A · dA[B,y,g]
    #
    # So: d²ω/(dR_Ax dR_By) = -2·wgp/A · dA[B,y,g] · dA[A,x,g] + wgp · d²A[A,B,x,y,g]
    #
    # Term (4) contribution: d²ω · (-diagd_dm)

    # Sub-term 4a: (-2·wgp/A · dA[B,y] · dA[A,x]) · (-diagd_dm)
    #            = 2·wgp/A · dA[A,x] · dA[B,y] · diagd_dm
    d2e += np.einsum('g,Axg,Byg,g->ABxyg',
                     2.0 * wgrad_prefs / areas, dareas, dareas, diagd_dm,
                     optimize=True)

    # Sub-term 4b: (wgp · d²A[A,B,x,y,g]) · (-diagd_dm)
    #            = -wgp · d²A · diagd_dm
    d2e -= np.einsum('g,ABxyg,g->ABxyg', wgrad_prefs, d2A, diagd_dm,
                     optimize=True)

    # Term (5): dω/dR_Ax · dω/dR_By · Tr[D · ∂²Gtilde/∂ω²]
    # ∂²Gtilde/∂ω² = <i|(r-Rg)⁴·exp(-ω|r-Rg|²)|j>
    # = Laplacian-of-Laplacian of g-type (l=4) integral
    #
    # (r-Rg)⁴ = [(r-Rg)²]² = Σ_αβ (r_α-Rg_α)²(r_β-Rg_β)²
    # In Cartesian l=4: need x⁴ + y⁴ + z⁴ + 2x²y² + 2x²z² + 2y²z²
    gmol_g4 = fakemol_for_gaussian(gost.grid_coords, widths, l=4)
    supermol_g4 = mol_obj + gmol_g4
    supermol_g4.cart = True
    slices_g4 = (0, mol_obj.nbas, 0, mol_obj.nbas,
                 mol_obj.nbas, mol_obj.nbas + gmol_g4.nbas)
    overlap3g = supermol_g4.intor('int3c1e', shls_slice=slices_g4
                                  ).reshape(nao_cart, nao_cart, ngrids, 15)
    if not mol_obj.cart:
        overlap3g = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3g, c2s, optimize=True)

    # Cartesian l=4 ordering (lexicographic in PySCF):
    # 0:xxxx, 1:xxxy, 2:xxxz, 3:xxyy, 4:xxyz, 5:xxzz,
    # 6:xyyy, 7:xyyz, 8:xyzz, 9:xzzz, 10:yyyy, 11:yyyz, 12:yyzz, 13:yzzz, 14:zzzz
    #
    # (r-Rg)⁴ = x⁴ + y⁴ + z⁴ + 2x²y² + 2x²z² + 2y²z²
    # = comp[0] + comp[10] + comp[14] + 2*comp[3] + 2*comp[5] + 2*comp[12]
    gtype_trace = (overlap3g[:, :, :, 0] + overlap3g[:, :, :, 10]
                   + overlap3g[:, :, :, 14]
                   + 2.0 * overlap3g[:, :, :, 3]
                   + 2.0 * overlap3g[:, :, :, 5]
                   + 2.0 * overlap3g[:, :, :, 12])
    gtype_dm = np.einsum('ijg,ij->g', gtype_trace, dm, optimize=True)
    del overlap3g, gtype_trace

    # Term (5): dω/dR_Ax · dω/dR_By · gtype_dm
    # = wgp·dA[A,x] · wgp·dA[B,y] · gtype_dm
    d2e += np.einsum('g,Axg,g,Byg,g->ABxyg',
                     wgrad_prefs, dareas, wgrad_prefs, dareas, gtype_dm,
                     optimize=True)

    return d2e


def test_d2e(gost, dm, mol, step=1e-4, label=""):
    """Test d²e_g/(dR dR) against finite differences."""
    print(f"Computing d²e_g fdiff ({label})...")
    d2e_fd = compute_d2e_fd(gost, dm, mol, step=step)

    print(f"Computing d²e_g analytical ({label})...")
    d2e_ana = compute_d2e_analytical(gost, dm, mol)

    natm = mol.natm
    ngrids = gost.n_gaussian

    err_full = np.max(np.abs(d2e_ana - d2e_fd))
    max_val = np.max(np.abs(d2e_fd))

    print(f"\n=== d²e_g Validation ({label}) ===")
    print(f"  shape: ({natm}, {natm}, 3, 3, {ngrids})")
    print(f"  max |d2e_fd|:  {max_val:.2e}")
    print(f"  max error: {err_full:.2e}")
    print(f"  relative error: {err_full/max_val:.2e}")

    if err_full / max_val < 1e-5:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        residual = d2e_fd - d2e_ana
        for A in range(natm):
            for B in range(natm):
                r = np.max(np.abs(residual[A, B]))
                if r > 1e-8:
                    print(f"    residual[{A},{B}]: {r:.2e}")

    sym_err = np.max(np.abs(d2e_fd - d2e_fd.transpose(1, 0, 3, 2, 4)))
    print(f"  fdiff symmetry: {sym_err:.2e}")

    return d2e_ana, d2e_fd


if __name__ == '__main__':
    print("=" * 60)
    print("H2/sto-3g")
    print("=" * 60)
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")
    test_d2e(gost, dm, mol, label="H2/sto-3g")

    print("\n" + "=" * 60)
    print("H2O/cc-pVDZ")
    print("=" * 60)
    gost, dm, mol = make_h2o_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")
    test_d2e(gost, dm, mol, label="H2O/cc-pVDZ")
