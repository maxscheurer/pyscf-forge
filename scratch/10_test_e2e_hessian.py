#!/usr/bin/env python
"""
Test 10: End-to-end Hessian validation.

Compares mf.Hessian().kernel() (the full PySCF analytical Hessian, including
orbital response) against central finite differences of the full SCF gradient
at displaced geometries.

This is the definitive test: no monkey-patching, no scratch imports.
Just the PySCF API.
"""
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP


def numerical_hessian(mol, gost_opts, step=1e-4):
    """Full numerical Hessian via central fdiff of SCF+GOSTSHYP gradient."""
    natm = mol.natm
    coords0 = mol.atom_coords().copy()
    hess = np.zeros((natm, 3, natm, 3))

    for B in range(natm):
        for y in range(3):
            for sign, s in [(+1, step), (-1, -step)]:
                coords = coords0.copy()
                coords[B, y] += s
                mol_d = mol.copy()
                mol_d.set_geom_(coords, unit='Bohr')
                gost_d = GOSTSHYP(mol_d, options=gost_opts)
                mf_d = scf.RHF(mol_d).GOSTSHYP(solvent_obj=gost_d)
                mf_d.conv_tol = 1e-12
                mf_d.verbose = 0
                mf_d.kernel()
                grad_d = mf_d.nuc_grad_method()
                grad_d.verbose = 0
                g = grad_d.kernel()
                if sign == 1:
                    grad_p = g
                else:
                    grad_m = g
            hess[:, :, B, y] = (grad_p - grad_m) / (2 * step)

    hess = hess.transpose(0, 2, 1, 3)
    return 0.5 * (hess + hess.transpose(1, 0, 3, 2))


def test_e2e(atom, basis, gost_opts, label):
    """Run end-to-end test for a given molecule."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    gost = GOSTSHYP(mol, options=gost_opts)
    mf = scf.RHF(mol).GOSTSHYP(solvent_obj=gost)
    mf.conv_tol = 1e-12
    mf.kernel()

    gost = mf.with_solvent
    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'  E = {mf.e_tot:.10f}')
    print(f'  {mol.natm} atoms, {gost.n_gaussian} grid pts')
    print(f'  min area = {gost.areas.min():.3e}')
    print(f'{"="*60}')

    print('  Computing analytical Hessian via mf.Hessian().kernel()...')
    hessobj = mf.Hessian()
    hess_ana = hessobj.kernel()

    print('  Computing numerical Hessian via fdiff of SCF gradient...')
    hess_num = numerical_hessian(mol, gost_opts)

    err = np.max(np.abs(hess_ana - hess_num))
    ref = np.max(np.abs(hess_num))

    print(f'\n  max |H_num|:  {ref:.3e}')
    print(f'  max |error|:  {err:.3e}')
    print(f'  relative:     {err/ref:.2e}')

    if err < 1e-5:
        print('  PASS ✓')
    elif err / ref < 1e-4:
        print(f'  PASS ✓ (abs err {err:.1e} but rel err OK)')
    else:
        print('  FAIL ✗')
        for A in range(mol.natm):
            for B in range(A, mol.natm):
                blk_err = np.max(np.abs(hess_ana[A, B] - hess_num[A, B]))
                if blk_err > 1e-6:
                    print(f'    [{A},{B}]: {blk_err:.3e}')

    return hess_ana, hess_num


if __name__ == '__main__':
    opts_vdw = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}

    test_e2e('H 0 0 0; H 0 0 0.74', 'sto-3g', opts_vdw,
             'H2/sto-3g/vdw')

    test_e2e('N 0 0 0; N 0 0 1.098', 'cc-pVDZ', opts_vdw,
             'N2/cc-pVDZ/vdw')
