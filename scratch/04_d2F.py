#!/usr/bin/env python
"""
Test 4: Validate d²F_g/(dR_Ax dR_By) — second derivative of Fhat trace.

This is needed for Hessian term H9: -A·e/F² · d²F/(dR_Ax dR_By)

Same structure as d²e but with p-type fakemol contracted with normals.
For vdw cavity, normals are geometry-independent (Lebedev directions).

We validate by finite-differencing dF_g/dR_Ax at displaced geometries.
"""

import numpy as np
from pyscf import gto
from pyscf.solvent.gostshyp import GOSTSHYP, fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A

from common import (make_h2_system, compute_scalar_traces,
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


def compute_d2F_analytical(gost, dm, mol):
    """Compute d²F_g/(dR_Ax dR_By) analytically.

    Returns shape (natm, natm, 3, 3, ngrids).

    Same structure as d²e but with p-type integrals contracted with normals.
    """
    mol_obj = gost.mol
    nao = mol_obj.nao_nr()
    nao_cart = mol_obj.nao_nr(cart=True)
    natm = mol_obj.natm
    ngrids = gost.n_gaussian
    aoslice = mol_obj.aoslice_by_atom()

    widths = gost.widths
    normals = gost.surface_normals

    if not mol_obj.cart:
        c2s = mol_obj.cart2sph_coeff(normalized='sp')

    d2F = np.zeros((natm, natm, 3, 3, ngrids))

    # Build p-type fakemol
    gmol_p = fakemol_for_gaussian(gost.grid_coords, widths, l=1,
                                  coeffs=2.0 * widths)
    supermol = mol_obj + gmol_p
    slices_p = (0, mol_obj.nbas, 0, mol_obj.nbas,
                mol_obj.nbas, mol_obj.nbas + gmol_p.nbas)

    # === Position × Position terms ===

    # --- ipip1: ∂²/∂R_bra² ---
    ipip1_raw = supermol.intor('int3c1e_ipip1', shls_slice=slices_p
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip1 = np.einsum('xijgc,gc->xijg', ipip1_raw, normals, optimize=True)
    del ipip1_raw

    ipip1_dm = np.einsum('xijg,ij->xig', ipip1, dm, optimize=True)
    ipip1_dm += np.einsum('xijg,ji->xig', ipip1, dm, optimize=True)
    for A in range(natm):
        p0, p1 = aoslice[A, 2], aoslice[A, 3]
        contrib = np.sum(ipip1_dm[:, p0:p1, :], axis=1)
        d2F[A, A, :, :, :] -= contrib.reshape(3, 3, ngrids)
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
            d2F[A, B, :, :, :] -= trace_block.reshape(3, 3, ngrids)
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
            d2F[A, B, :, :, g] -= contrib_A[:, g].reshape(3, 3)
    del ip1ip2_dm

    # --- aux × bra/ket (reverse of ip1ip2) ---
    ip1ip2_dm_t = np.einsum('xijg,ij->xig', ip1ip2, dm, optimize=True)
    ip1ip2_dm_t += np.einsum('xijg,ji->xig', ip1ip2, dm, optimize=True)
    for g in range(ngrids):
        A = gost.atom_idx[g]
        for B in range(natm):
            p0, p1 = aoslice[B, 2], aoslice[B, 3]
            trace_ig = np.sum(ip1ip2_dm_t[:, p0:p1, g], axis=1)
            d2F[A, B, :, :, g] -= trace_ig.reshape(3, 3).T
    del ip1ip2, ip1ip2_dm_t

    # --- ipip2: ∂²/∂R_aux² ---
    ipip2_raw = supermol.intor('int3c1e_ipip2', shls_slice=slices_p
                               ).reshape(9, nao, nao, ngrids, 3)
    ipip2 = np.einsum('xijgc,gc->xijg', ipip2_raw, normals, optimize=True)
    del ipip2_raw

    ipip2_dm = np.einsum('xijg,ij->xg', ipip2, dm, optimize=True)
    for g in range(ngrids):
        B = gost.atom_idx[g]
        d2F[B, B, :, :, g] -= ipip2_dm[:, g].reshape(3, 3)
    del ipip2, ipip2_dm

    # === Width response terms ===
    # TODO: implement width × position and width × width terms for Fhat
    # These involve f-type ip1 integrals and higher angular momentum

    return d2F


def test_d2F(gost, dm, mol, step=1e-4):
    """Test d²F_g/(dR dR) against finite differences."""
    print("Computing d²F_g fdiff (may take a moment)...")
    d2F_fd = compute_d2F_fd(gost, dm, mol, step=step)

    print("Computing d²F_g analytical (position-position terms only)...")
    d2F_ana = compute_d2F_analytical(gost, dm, mol)

    natm = mol.natm
    ngrids = gost.n_gaussian

    err_full = np.max(np.abs(d2F_ana - d2F_fd))
    max_val = np.max(np.abs(d2F_fd))

    print(f"\n=== d²F_g Validation ===")
    print(f"  shape: ({natm}, {natm}, 3, 3, {ngrids})")
    print(f"  max |d2F_fd|:  {max_val:.2e}")
    print(f"  max error (positions only): {err_full:.2e}")
    print(f"  relative error: {err_full/max_val:.2e}")

    if err_full < 1e-5:
        print("  PASS ✓ (position terms sufficient)")
    else:
        print("  (residual = width response terms, to be implemented)")
        residual = d2F_fd - d2F_ana
        print(f"  residual max: {np.max(np.abs(residual)):.2e}")
        for A in range(natm):
            for B in range(natm):
                r = np.max(np.abs(residual[A, B]))
                if r > 1e-8:
                    print(f"    residual[{A},{B}]: {r:.2e}")

    sym_err = np.max(np.abs(d2F_fd - d2F_fd.transpose(1, 0, 3, 2, 4)))
    print(f"  fdiff symmetry: {sym_err:.2e}")

    return d2F_ana, d2F_fd


if __name__ == '__main__':
    print("Setting up H2/sto-3g...")
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")

    test_d2F(gost, dm, mol)
