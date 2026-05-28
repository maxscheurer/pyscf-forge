#!/usr/bin/env python
"""
Test 1: Validate first-derivative scalar traces de_g/dR and dF_g/dR.

These are the building blocks for the Hessian cross-terms (H2-H4, H6-H8, H10).
Validates by finite-differencing e_g(R) and F_g(R).
"""

import numpy as np
from common import (make_h2_system, compute_scalar_traces,
                    compute_eg_Fg_at_geom, get_gost_options)


def test_scalar_traces(gost, dm, mol, step=1e-4):
    natm = mol.natm
    ngrids = gost.n_gaussian
    coords0 = mol.atom_coords().copy()
    opts = get_gost_options(gost)

    # Analytical
    dg_ana, dF_ana = compute_scalar_traces(gost, dm)

    # Finite differences
    dg_fd = np.zeros((natm, 3, ngrids))
    dF_fd = np.zeros((natm, 3, ngrids))

    for A in range(natm):
        for x in range(3):
            coords_p = coords0.copy()
            coords_p[A, x] += step
            eg_p, Fg_p = compute_eg_Fg_at_geom(mol, dm, opts, coords_p)

            coords_m = coords0.copy()
            coords_m[A, x] -= step
            eg_m, Fg_m = compute_eg_Fg_at_geom(mol, dm, opts, coords_m)

            dg_fd[A, x, :] = (eg_p - eg_m) / (2.0 * step)
            dF_fd[A, x, :] = (Fg_p - Fg_m) / (2.0 * step)

    # Compare
    dg_err = np.max(np.abs(dg_ana - dg_fd))
    dF_err = np.max(np.abs(dF_ana - dF_fd))

    print("=== Scalar Trace Validation (de_g/dR, dF_g/dR) ===")
    print(f"  de_g/dR max error: {dg_err:.2e}  (vs fdiff step={step})")
    print(f"  dF_g/dR max error: {dF_err:.2e}  (vs fdiff step={step})")

    if dg_err < 1e-6:
        print("  de_g/dR: PASS ✓")
    else:
        print("  de_g/dR: FAIL ✗")
        # Show worst element
        idx = np.unravel_index(np.argmax(np.abs(dg_ana - dg_fd)), dg_ana.shape)
        print(f"    worst at (A={idx[0]}, x={idx[1]}, g={idx[2]}): "
              f"ana={dg_ana[idx]:.10e}, fd={dg_fd[idx]:.10e}")

    if dF_err < 1e-6:
        print("  dF_g/dR: PASS ✓")
    else:
        print("  dF_g/dR: FAIL ✗")
        idx = np.unravel_index(np.argmax(np.abs(dF_ana - dF_fd)), dF_ana.shape)
        print(f"    worst at (A={idx[0]}, x={idx[1]}, g={idx[2]}): "
              f"ana={dF_ana[idx]:.10e}, fd={dF_fd[idx]:.10e}")

    return dg_err, dF_err


if __name__ == '__main__':
    print("Setting up H2/sto-3g...")
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")

    test_scalar_traces(gost, dm, mol)
