#!/usr/bin/env python
"""
Test 3: Validate d²e_g/(dR_Ax dR_By) — second derivative of Gtilde trace.

This is the hardest new computation for the Hessian (term H5).
d²e_g/(dR_Ax dR_By) = Tr[D · d²Gtilde_g/(dR_Ax dR_By)]

We validate by finite-differencing de_g/dR_Ax (from compute_scalar_traces)
at displaced geometries.

Sub-terms:
  - Position × Position: ipip1, ipvip1, ip1ip2, ipip2
  - Position × Width: ip1 derivatives of d-type integrals
  - Width × Width: g-type (l=4) integrals

Strategy: implement incrementally, comparing partial results to fdiff.
"""

import numpy as np
from pyscf import gto
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A

from common import (make_h2_system, compute_scalar_traces,
                    get_gost_options)


def compute_d2e_fd(gost, dm, mol, step=1e-4):
    """Compute d²e_g/(dR_Ax dR_By) by finite-differencing de_g/dR.

    Returns shape (natm, natm, 3, 3, ngrids).
    d2e[A, B, x, y, g] = d(de_g/dR_Ax)/dR_By
    """
    natm = mol.natm
    ngrids = gost.n_gaussian
    coords0 = mol.atom_coords().copy()
    opts = get_gost_options(gost)

    d2e_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            # Plus displacement
            coords_p = coords0.copy()
            coords_p[B, y] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=opts)
            gost_p.kernel(dm)
            dg_p, _ = compute_scalar_traces(gost_p, dm)

            # Minus displacement
            coords_m = coords0.copy()
            coords_m[B, y] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=opts)
            gost_m.kernel(dm)
            dg_m, _ = compute_scalar_traces(gost_m, dm)

            # d²e[A, B, x, y, g] = (de_p[A,x,g] - de_m[A,x,g]) / (2*step)
            d2e_fd[:, B, :, y, :] = (dg_p - dg_m) / (2.0 * step)

    return d2e_fd


def compute_d2e_analytical(gost, dm, mol):
    """Compute d²e_g/(dR_Ax dR_By) analytically.

    Returns shape (natm, natm, 3, 3, ngrids).

    This is the function to implement term-by-term.
    Start with position-position, then add width terms.
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

    # Get area derivatives for width terms
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dareas = dareas_raw.transpose(1, 2, 0)  # (natm, 3, ngrids)
        _, d2A = get_d2F_d2A(gost.surface_dict)
    # d2A shape: (natm, natm, 3, 3, ngrids)

    d2e = np.zeros((natm, natm, 3, 3, ngrids))

    # === Part 1: Position × Position second derivatives ===
    # Build fakemols
    gmol_s = fakemol_for_gaussian(gost.grid_coords, widths)
    supermol = mol_obj + gmol_s
    slices_s = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_s.nbas)
    slices_sg = (mol_obj.nbas, mol_obj.nbas + gmol_s.nbas,
                 0, mol_obj.nbas, 0, mol_obj.nbas)

    # --- ipip1: ∂²/∂R_bra² (both on same bra atom) ---
    # Shape: (9, nao, nao, ngrids) → 9 = (xx,xy,xz,yx,yy,yz,zx,zy,zz)
    ipip1 = supermol.intor('int3c1e_ipip1', shls_slice=slices_s)

    # Trace with D: Σ_j (D_ij + D_ji) * ipip1[xy,i,j,g]
    # For A=B, sum over i∈A:
    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        # ipip1 contributes to d²e[A, A, x, y, g]
        contrib = np.sum(ipip1_dm[:, p0:p1, :], axis=1)  # (9, ngrids)
        d2e[A, A, :, :, :] -= contrib.reshape(3, 3, ngrids)
    del ipip1, ipip1_dm

    # --- ipvip1: ∂²/(∂R_bra ∂R_ket) (bra on A, ket on B) ---
    ipvip1 = supermol.intor('int3c1e_ipvip1', shls_slice=slices_s)

    # Trace: Σ_{i∈A, j∈B} D_ij * ipvip1[xy,i,j,g]
    # Note: ipvip1 is NOT symmetric in i,j — careful with bra/ket roles
    # The first ip acts on center 1 (bra=i), the second ip acts on center 2 (ket=j)
    for A in range(natm):
        p0_A, p1_A = aoslice[A, 2], aoslice[A, 3]
        for B in range(natm):
            p0_B, p1_B = aoslice[B, 2], aoslice[B, 3]
            # D_ij * ipvip1_xy[i∈A, j∈B, g] + D_ji * ipvip1_xy[i∈A, j∈B, g]
            # The bra-ket symmetry: for <i|g|j>, bra deriv on i, ket deriv on j
            # Also need <j|g|i> contribution: bra deriv on j∈A... wait.
            #
            # Full contribution to d²e[A,B,x,y]:
            # From bra=A, ket=B: Σ_{i∈A,j∈B} D_ij * ipvip1[xy,i,j,g]
            # From bra=B, ket=A (using ipvip1 is bra×ket):
            #   Σ_{i∈B,j∈A} D_ij * ipvip1[yx,i,j,g] → goes into d²e[B,A,y,x]
            #
            # But we also have the D^T contribution from the <j|g|i> integral:
            # Σ_{i∈A,j∈B} D_ji * ipvip1[xy,i,j,g]
            # This accounts for the ket-function-on-A contribution to de/dR_A
            # crossed with bra-function-on-B... hmm.
            #
            # Let me be precise: de/dR_Ax uses:
            #   -Σ_{i∈A} (D+D^T)[i,:] · ip1[x,i,:,g]
            # Taking d/dR_By of ip1[x,i,j,g] when j∈B gives ipvip1[xy,i,j,g]
            # So: d²e[A,B,x,y] gets -Σ_{i∈A,j∈B} (D+D^T)_ij * ipvip1[xy,i,j,g]
            block = ipvip1[:, p0_A:p1_A, p0_B:p1_B, :]
            trace_block = (np.einsum('xijg,ij->xg', block, dm[p0_A:p1_A, p0_B:p1_B], optimize=True)
                           + np.einsum('xijg,ji->xg', block, dm[p0_B:p1_B, p0_A:p1_A], optimize=True))
            d2e[A, B, :, :, :] -= trace_block.reshape(3, 3, ngrids)
    del ipvip1

    # --- ip1ip2: ∂²/(∂R_bra ∂R_aux) (bra on A, aux on B) ---
    ip1ip2 = supermol.intor('int3c1e_ip1ip2', shls_slice=slices_s)

    # Trace: -Σ_{i∈A} (D+D^T)_i. * ip1ip2[xy,i,j,g on B]
    ip1ip2_dm = np.einsum('xijg,ij->xig', ip1ip2, dm, optimize=True)
    ip1ip2_dm += np.einsum('xijg,ji->xig', ip1ip2, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib_A = np.sum(ip1ip2_dm[:, p0:p1, :], axis=1)  # (9, ngrids)
        # This contributes to d²e[A, B, x, y, g] where B owns grid point g
        for g in range(ngrids):
            B = gost.atom_idx[g]
            d2e[A, B, :, :, g] -= contrib_A[:, g].reshape(3, 3)
    del ip1ip2, ip1ip2_dm

    # --- ipip2: ∂²/∂R_aux² (both on aux center, same atom) ---
    ipip2 = supermol.intor('int3c1e_ipip2', shls_slice=slices_s)

    # Trace with D:
    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    # Contributes to d²e[B, B, x, y, g] where B owns g
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2e[B, B, :, :, g] -= ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    # --- aux × bra/ket cross: ∂²/(∂R_aux ∂R_bra) ---
    # Need d/dR_By(bra/ket) of [aux-ip1 contribution from atom A owning g]
    # = Σ_{i,j} (D_ij) * ∂²I/(∂R_g_x ∂R_i_y)  for i∈B
    # Use ip1ip2 but with swapped roles:
    # ip1ip2[xy] = ∂²I/(∂R_bra_x ∂R_aux_y) ... we need (∂R_aux_x ∂R_bra_y)
    # By symmetry of mixed partials: ∂²I/(∂R_aux_x ∂R_bra_y) = ∂²I/(∂R_bra_y ∂R_aux_x)
    # = ip1ip2 with x↔y components, i.e., ip1ip2.reshape(3,3,...)[y,x,...]
    # Actually ip1ip2[comp] with comp = 3*x + y, so swap is comp = 3*y + x
    # i.e., transpose the 3×3 block
    #
    # Wait — we already covered "bra on A, aux on B" above (ip1ip2 block).
    # Here we need "aux on A, bra/ket on B" — that's d²e[A,B] where A owns g.
    # This is the TRANSPOSE of the ip1ip2 contribution:
    # d²e[A, B, x, y, g] with A owning g, and the second deriv on bra/ket of B
    # = ∂²I/(∂R_g_x ∂R_j_y) summed over j∈B
    # Use: ∂²/(∂R_aux ∂R_bra) = ip1ip2.T (swap the two derivative indices)

    # Re-compute ip1ip2 for the reverse contribution
    ip1ip2 = supermol.intor('int3c1e_ip1ip2', shls_slice=slices_s)
    ip1ip2_dm_t = np.einsum('xijg,ij->xig', ip1ip2, dm, optimize=True)
    ip1ip2_dm_t += np.einsum('xijg,ji->xig', ip1ip2, dm, optimize=True)
    # ip1ip2_dm_t[comp, i, g] where comp = 3*x_bra + y_aux
    # For aux-on-A (x), bra/ket-on-B (y):
    # d²e[A(owns g), B(owns i), x_aux, y_bra, g]
    # = ip1ip2_dm_t[3*y_bra + x_aux, i∈B, g where atom_idx[g]=A]
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            # Sum over i∈B
            trace_ig = np.sum(ip1ip2_dm_t[:, p0:p1, g], axis=1)  # (9,)
            # trace_ig[3*x_bra + y_aux] → need to map to d2e[A,B,x_aux,y_bra]
            # ip1ip2 has comp ordering (x_bra, y_aux) → (3*x + y)
            # We want d2e[A, B, x_aux, y_bra] so transpose the 3×3:
            d2e[A, B, :, :, g] -= trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm_t

    # === Part 2: Width response terms ===
    # TODO: implement width × position and width × width terms
    # These involve:
    # - d-type ip1 integrals (position derivative of ∂Gtilde/∂ω)
    # - g-type (l=4) integrals (∂²Gtilde/∂ω²)
    # - d²ω/(dR dR) terms from area second derivatives

    # For now, return only the position-position part
    # The residual vs fdiff shows what the width terms contribute

    return d2e


def test_d2e(gost, dm, mol, step=1e-4):
    """Test d²e_g/(dR dR) against finite differences."""
    print("Computing d²e_g fdiff (may take a moment)...")
    d2e_fd = compute_d2e_fd(gost, dm, mol, step=step)

    print("Computing d²e_g analytical (position-position terms only)...")
    d2e_ana = compute_d2e_analytical(gost, dm, mol)

    natm = mol.natm
    ngrids = gost.n_gaussian

    # Full comparison
    err_full = np.max(np.abs(d2e_ana - d2e_fd))
    max_val = np.max(np.abs(d2e_fd))

    print(f"\n=== d²e_g Validation ===")
    print(f"  shape: ({natm}, {natm}, 3, 3, {ngrids})")
    print(f"  max |d2e_fd|:  {max_val:.2e}")
    print(f"  max error (positions only): {err_full:.2e}")
    print(f"  relative error: {err_full/max_val:.2e}")

    if err_full < 1e-5:
        print("  PASS ✓ (position terms sufficient)")
    else:
        print("  (residual = width response terms, to be implemented)")
        # Show the residual structure
        residual = d2e_fd - d2e_ana
        print(f"  residual max: {np.max(np.abs(residual)):.2e}")
        # Per-atom-pair breakdown
        for A in range(natm):
            for B in range(natm):
                r = np.max(np.abs(residual[A, B]))
                if r > 1e-8:
                    print(f"    residual[{A},{B}]: {r:.2e}")

    # Check symmetry of fdiff result
    sym_err = np.max(np.abs(d2e_fd - d2e_fd.transpose(1, 0, 3, 2, 4)))
    print(f"  fdiff symmetry d2e[A,B,x,y]=d2e[B,A,y,x]: {sym_err:.2e}")

    return d2e_ana, d2e_fd


if __name__ == '__main__':
    print("Setting up H2/sto-3g...")
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")

    test_d2e(gost, dm, mol)
