#!/usr/bin/env python
"""
Test 2: Validate d²A from get_d2F_d2A against fdiff of dA from get_dF_dA.

This tests Hessian term H1: d²A/(dR_Ax dR_By) * e_g / F_g

The surface area second derivative is provided by pyscf's PCM infrastructure.
We validate it by finite-differencing get_dF_dA at displaced geometries.
"""

import numpy as np
from pyscf import gto
from pyscf.solvent.pcm import gen_surface, modified_Bondi
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A

from common import make_h2_system, get_gost_options
from pyscf.solvent.gostshyp import GOSTSHYP


def compute_dA_at_geom(mol_template, coords, cavity, npoints, scaling_factor,
                       r_ext=None):
    """Compute dA/dR at a given geometry. Returns (natm, 3, ngrids)."""
    mol = mol_template.copy()
    mol.set_geom_(coords, unit='Bohr')

    rad = scaling_factor * modified_Bondi
    if cavity == 'vdw/occ':
        rad_outer = rad + r_ext
        surface = gen_surface(mol, ng=npoints, rad=rad_outer)
    else:
        surface = gen_surface(mol, ng=npoints, rad=rad)

    _, dA = get_dF_dA(surface)
    # dA shape from get_dF_dA: (ngrids, natm, 3)
    return dA.transpose(1, 2, 0)  # → (natm, 3, ngrids)


def test_d2A(gost, dm, mol, step=1e-4):
    natm = mol.natm
    coords0 = mol.atom_coords().copy()
    opts = get_gost_options(gost)

    # Get the surface dict for analytical d²A
    if gost._outer_surface_dict is not None:
        surface = gost._outer_surface_dict
    else:
        surface = gost.surface_dict

    # Analytical d²A: shape (natm, natm, 3, 3, ngrids)
    _, d2A_ana = get_d2F_d2A(surface)
    ngrids = d2A_ana.shape[-1]

    # Finite-difference d²A by differencing dA
    d2A_fd = np.zeros((natm, natm, 3, 3, ngrids))

    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy()
            coords_p[B, y] += step
            dA_p = compute_dA_at_geom(mol, coords_p, opts['cavity'],
                                      opts['npoints'], opts['scaling_factor'],
                                      opts.get('r_ext'))

            coords_m = coords0.copy()
            coords_m[B, y] -= step
            dA_m = compute_dA_at_geom(mol, coords_m, opts['cavity'],
                                      opts['npoints'], opts['scaling_factor'],
                                      opts.get('r_ext'))

            # d²A[A, B, x, y, g] = d(dA[A,x,g])/dR_By
            d2A_fd[:, B, :, y, :] = (dA_p - dA_m) / (2.0 * step)

    # Compare
    err = np.max(np.abs(d2A_ana - d2A_fd))

    print("=== d²A Validation (get_d2F_d2A vs fdiff of get_dF_dA) ===")
    print(f"  max error: {err:.2e}  (step={step})")
    print(f"  max |d2A|: {np.max(np.abs(d2A_ana)):.2e}")

    if err < 1e-5:
        print("  PASS ✓")
    else:
        print("  FAIL ✗")
        idx = np.unravel_index(np.argmax(np.abs(d2A_ana - d2A_fd)), d2A_ana.shape)
        print(f"    worst at A={idx[0]}, B={idx[1]}, x={idx[2]}, y={idx[3]}, g={idx[4]}")
        print(f"    ana={d2A_ana[idx]:.10e}, fd={d2A_fd[idx]:.10e}")

    # Also check symmetry: d²A[A,B,x,y,g] = d²A[B,A,y,x,g]
    sym_err = np.max(np.abs(d2A_ana - d2A_ana.transpose(1, 0, 3, 2, 4)))
    print(f"  symmetry error: {sym_err:.2e}")

    return err


if __name__ == '__main__':
    print("Setting up H2/sto-3g (vdw cavity)...")
    gost, dm, mol = make_h2_system(cavity='vdw')
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points\n")

    test_d2A(gost, dm, mol)
