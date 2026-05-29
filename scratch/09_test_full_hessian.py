#!/usr/bin/env python
"""
Test 9: Full analytical Hessian on molecules without small-area grid points.

SF6 (octahedral, 7 atoms, 422 grid points) has min_area = 5.1e-3,
so the d²F conditioning issue doesn't arise. This provides a clean
end-to-end validation of the fully analytical Hessian.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP
from importlib import import_module

_assembly = import_module('05_hess_assembly')
assemble_hessian = _assembly.assemble_hessian

_d2e = import_module('03_d2e')
_d2F = import_module('04_d2F')


def make_sf6_system(cavity='vdw', npoints=110, scaling_factor=1.2,
                    pressure_mpa=50_000):
    """SF6/cc-pVDZ — octahedral, 7 atoms, no small-area grid points."""
    mol = gto.M(
        atom='''S  0.000  0.000  0.000
                F  1.560  0.000  0.000
                F -1.560  0.000  0.000
                F  0.000  1.560  0.000
                F  0.000 -1.560  0.000
                F  0.000  0.000  1.560
                F  0.000  0.000 -1.560''',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    options = {
        'cavity': cavity,
        'pressure_mpa': pressure_mpa,
        'npoints': npoints,
        'scaling_factor': scaling_factor,
    }
    gost = GOSTSHYP(mol, options=options)
    gost.kernel(dm)
    return gost, dm, mol


def make_n2_system(cavity='vdw', npoints=110, scaling_factor=1.2,
                   pressure_mpa=50_000):
    """N2/cc-pVDZ — diatomic, clean baseline."""
    mol = gto.M(
        atom='N 0 0 0; N 0 0 1.098',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    options = {
        'cavity': cavity,
        'pressure_mpa': pressure_mpa,
        'npoints': npoints,
        'scaling_factor': scaling_factor,
    }
    gost = GOSTSHYP(mol, options=options)
    gost.kernel(dm)
    return gost, dm, mol


def test_full_hessian(gost, dm, mol, label=""):
    """Test fully analytical Hessian against hess_fd."""
    print(f"\n{'='*60}")
    print(f"Full Analytical Hessian: {label}")
    print(f"  {mol.natm} atoms, {gost.n_gaussian} grid points")
    print(f"  min area = {gost.areas.min():.3e}, "
          f"pts with area<1e-3 = {(gost.areas < 1e-3).sum()}")
    print(f"{'='*60}")

    print("  Computing hess_fd reference...")
    hess_fd = gost.hess_fd(dm, step=1e-4)

    print("  Computing analytical d²e...")
    d2e = _d2e.compute_d2e_analytical(gost, dm, mol)

    print("  Computing analytical d²F...")
    d2F = _d2F.compute_d2F_analytical(gost, dm, mol)

    print("  Assembling Hessian...")
    hess_ana = assemble_hessian(gost, dm, mol, d2e=d2e, d2F=d2F)

    err = np.max(np.abs(hess_ana - hess_fd))
    ref = np.max(np.abs(hess_fd))

    print(f"\n  max |hess_fd|:  {ref:.3e}")
    print(f"  max |error|:    {err:.3e}")

    if err < 1e-5:
        print("  PASS ✓")
    else:
        print(f"  err/ref = {err/ref:.2e}")
        # Check if it's the d²F conditioning issue
        d2F_fd = _d2F.compute_d2F_fd(gost, dm, mol)
        hess_with_fd_d2F = assemble_hessian(gost, dm, mol, d2e=d2e, d2F=d2F_fd)
        err_with_fd = np.max(np.abs(hess_with_fd_d2F - hess_fd))
        print(f"  error with fdiff d²F: {err_with_fd:.3e}")
        if err_with_fd < 1e-5:
            print("  → Assembly correct; error is from d²F boundary points")
        else:
            print("  → Other issue")

    # Symmetry check
    sym = np.max(np.abs(hess_ana - hess_ana.transpose(1, 0, 3, 2)))
    print(f"  symmetry: {sym:.3e}")

    return hess_ana, hess_fd


if __name__ == '__main__':
    print("N2/cc-pVDZ (clean diatomic baseline)")
    gost, dm, mol = make_n2_system()
    test_full_hessian(gost, dm, mol, "N2/cc-pVDZ")

    print("\n\nSF6/cc-pVDZ (7-atom octahedral, no small-area points)")
    gost, dm, mol = make_sf6_system()
    test_full_hessian(gost, dm, mol, "SF6/cc-pVDZ")
