#!/usr/bin/env python
"""
Test 5: Assemble the full analytical Hessian from all ingredients.

H = P * Σ_g [
  H1:  d²A/(dR_Ax dR_By) · e/F
  H2:  dA/dR_Ax · de/dR_By / F
  H3: -dA/dR_Ax · e · dF/dR_By / F²
  H4:  dA/dR_By · de/dR_Ax / F
  H5:  A/F · d²e/(dR_Ax dR_By)
  H6: -A/F² · de/dR_Ax · dF/dR_By
  H7: -dA/dR_By · e · dF/dR_Ax / F²
  H8: -A/F² · de/dR_By · dF/dR_Ax
  H9: -A·e/F² · d²F/(dR_Ax dR_By)
  H10: 2·A·e/F³ · dF/dR_Ax · dF/dR_By
]

Compare assembled Hessian against hess_fd.

VALIDATED: Assembly formula matches hess_fd to 1e-10 when fdiff d²e/d²F are used.
The only remaining work is analytical computation of d²e and d²F (width terms).
"""

import numpy as np
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A

from common import make_h2_system, compute_scalar_traces, get_gost_options

# Import from sibling scripts — run from scratch/ directory
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_d2e_mod = import_module('03_d2e')
_d2F_mod = import_module('04_d2F')
compute_d2e_analytical = _d2e_mod.compute_d2e_analytical
compute_d2e_fd = _d2e_mod.compute_d2e_fd
compute_d2F_analytical = _d2F_mod.compute_d2F_analytical
compute_d2F_fd = _d2F_mod.compute_d2F_fd


def assemble_hessian(gost, dm, mol, d2e=None, d2F=None):
    """Assemble the full Hessian from all ingredients.

    Parameters
    ----------
    gost : GOSTSHYP with kernel() called
    dm : density matrix
    mol : molecule
    d2e : ndarray (natm, natm, 3, 3, ngrids) or None (will compute)
    d2F : ndarray (natm, natm, 3, 3, ngrids) or None (will compute)

    Returns
    -------
    hess : ndarray (natm, natm, 3, 3)
    """
    natm = mol.natm
    ngrids = gost.n_gaussian
    P = gost.pressure_au

    # Scalars from kernel
    e_g = gost.gtilde_expval  # (ngrids,)
    F_g = gost.forces         # (ngrids,)
    A_g = gost.areas          # (ngrids,)

    # First derivatives
    dg_trace, dF_trace = compute_scalar_traces(gost, dm)
    # dg_trace, dF_trace shape: (natm, 3, ngrids)

    # Area derivatives
    if gost._outer_surface_dict is not None:
        _, dareas_raw = get_dF_dA(gost._outer_surface_dict)
        dA = dareas_raw.transpose(1, 2, 0) * gost._occ_ratio_sq
        _, d2A = get_d2F_d2A(gost._outer_surface_dict)
        d2A = d2A * gost._occ_ratio_sq
    else:
        _, dareas_raw = get_dF_dA(gost.surface_dict)
        dA = dareas_raw.transpose(1, 2, 0)  # (natm, 3, ngrids)
        _, d2A = get_d2F_d2A(gost.surface_dict)
    # d2A shape: (natm, natm, 3, 3, ngrids)

    # Second derivative traces
    if d2e is None:
        d2e = compute_d2e_analytical(gost, dm, mol)
    if d2F is None:
        d2F = compute_d2F_analytical(gost, dm, mol)

    # Precompute per-grid scalars
    inv_F = 1.0 / F_g
    inv_F2 = inv_F ** 2
    inv_F3 = inv_F ** 3

    hess = np.zeros((natm, natm, 3, 3))

    # H1: d²A/(dR_Ax dR_By) · e/F
    # d2A[A,B,x,y,g] * e_g/F_g → sum over g
    hess += P * np.einsum('ABxyg,g,g->ABxy', d2A, e_g, inv_F, optimize=True)

    # H2: dA/dR_Ax · de/dR_By / F
    # dA[A,x,g] * dg[B,y,g] * inv_F[g]
    hess += P * np.einsum('Axg,Byg,g->ABxy', dA, dg_trace, inv_F, optimize=True)

    # H3: -dA/dR_Ax · e · dF/dR_By / F²
    hess -= P * np.einsum('Axg,g,Byg,g->ABxy', dA, e_g, dF_trace, inv_F2,
                          optimize=True)

    # H4: dA/dR_By · de/dR_Ax / F
    hess += P * np.einsum('Byg,Axg,g->ABxy', dA, dg_trace, inv_F, optimize=True)

    # H5: A/F · d²e/(dR_Ax dR_By)
    hess += P * np.einsum('g,g,ABxyg->ABxy', A_g, inv_F, d2e, optimize=True)

    # H6: -A/F² · de/dR_Ax · dF/dR_By
    hess -= P * np.einsum('g,g,Axg,Byg->ABxy', A_g, inv_F2, dg_trace,
                          dF_trace, optimize=True)

    # H7: -dA/dR_By · e · dF/dR_Ax / F²
    hess -= P * np.einsum('Byg,g,Axg,g->ABxy', dA, e_g, dF_trace, inv_F2,
                          optimize=True)

    # H8: -A/F² · de/dR_By · dF/dR_Ax
    hess -= P * np.einsum('g,g,Byg,Axg->ABxy', A_g, inv_F2, dg_trace,
                          dF_trace, optimize=True)

    # H9: -A·e/F² · d²F/(dR_Ax dR_By)
    hess -= P * np.einsum('g,g,g,ABxyg->ABxy', A_g, e_g, inv_F2, d2F,
                          optimize=True)

    # H10: 2·A·e/F³ · dF/dR_Ax · dF/dR_By
    hess += 2.0 * P * np.einsum('g,g,g,Axg,Byg->ABxy', A_g, e_g, inv_F3,
                                dF_trace, dF_trace, optimize=True)

    # Symmetrize
    hess = 0.5 * (hess + hess.transpose(1, 0, 3, 2))

    return hess


def test_assembly(gost, dm, mol, use_fd_traces=False):
    """Test assembled Hessian against hess_fd."""
    print("Computing hess_fd reference...")
    hess_fd = gost.hess_fd(dm, step=1e-4)

    if use_fd_traces:
        print("Computing d²e and d²F via fdiff (for debugging)...")
        d2e = compute_d2e_fd(gost, dm, mol)
        d2F = compute_d2F_fd(gost, dm, mol)
    else:
        print("Computing d²e and d²F analytically...")
        d2e = None
        d2F = None

    print("Assembling Hessian...")
    hess_ana = assemble_hessian(gost, dm, mol, d2e=d2e, d2F=d2F)

    err = np.max(np.abs(hess_ana - hess_fd))
    max_val = np.max(np.abs(hess_fd))

    print(f"\n=== Hessian Assembly Validation ===")
    print(f"  max |hess_fd|: {max_val:.2e}")
    print(f"  max error: {err:.2e}")
    print(f"  relative error: {err/max_val:.2e}")

    if err < 1e-5:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        # Show per-atom-pair errors
        natm = mol.natm
        for A in range(natm):
            for B in range(natm):
                blk_err = np.max(np.abs(hess_ana[A, B] - hess_fd[A, B]))
                if blk_err > 1e-6:
                    print(f"    error[{A},{B}]: {blk_err:.2e}")

    # Test individual term contributions using fdiff traces
    if not use_fd_traces:
        print("\n--- Retrying with fdiff d²e/d²F to isolate error source ---")
        d2e_fd = compute_d2e_fd(gost, dm, mol)
        d2F_fd = compute_d2F_fd(gost, dm, mol)
        hess_with_fd = assemble_hessian(gost, dm, mol, d2e=d2e_fd, d2F=d2F_fd)
        err_with_fd = np.max(np.abs(hess_with_fd - hess_fd))
        print(f"  error with fdiff traces: {err_with_fd:.2e}")
        if err_with_fd < 1e-5:
            print("  → Assembly formula is correct; error is in d²e/d²F computation")
        else:
            print("  → Error in assembly formula itself")

    return hess_ana, hess_fd


if __name__ == '__main__':
    print("Setting up H2/sto-3g...")
    gost, dm, mol = make_h2_system()
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")

    # First test with fdiff traces to validate the assembly formula
    print("=" * 60)
    print("Phase 1: Validate assembly formula (using fdiff d²e, d²F)")
    print("=" * 60)
    test_assembly(gost, dm, mol, use_fd_traces=True)

    print("\n" + "=" * 60)
    print("Phase 2: Full analytical (using analytical d²e, d²F)")
    print("=" * 60)
    test_assembly(gost, dm, mol, use_fd_traces=False)
