#!/usr/bin/env python
"""
Test 13: Validate analytical_grad_vmat (dV/dR at fixed D) against finite differences.

Compares the analytical derivative of the GOSTSHYP Fock matrix w.r.t. nuclear
coordinates against central finite differences of kernel() at displaced geometries.

Tests on:
  - N2/cc-pVDZ (baseline — should pass)
  - SF6/cc-pVDZ (the failing e2e case)
"""
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, analytical_grad_vmat


def numerical_grad_vmat(gost, dm, step=1e-5):
    """Finite-difference dV/dR at fixed dm."""
    mol = gost.mol
    natm = mol.natm
    nao = mol.nao_nr()
    coords0 = mol.atom_coords().copy()
    opts = get_opts(gost)

    dV_fd = np.zeros((natm, 3, nao, nao))
    for atm in range(natm):
        for x in range(3):
            coords_p = coords0.copy()
            coords_p[atm, x] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=opts)
            gost_p.kernel(dm)
            v_p = gost_p.v.copy()

            coords_m = coords0.copy()
            coords_m[atm, x] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=opts)
            gost_m.kernel(dm)
            v_m = gost_m.v.copy()

            dV_fd[atm, x] = (v_p - v_m) / (2.0 * step)

    return dV_fd


def get_opts(gost):
    """Extract options dict from a GOSTSHYP object."""
    return {
        'cavity': gost.cavity,
        'pressure_mpa': gost.pressure_mpa,
        'npoints': gost.npoints,
        'scaling_factor': gost.scaling_factor,
        'direct': gost.direct,
    }


def test_grad_vmat(atom, basis, gost_opts, label):
    """Test analytical_grad_vmat against fdiff."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    nao = mol.nao_nr()
    natm = mol.natm

    gost = GOSTSHYP(mol, options=gost_opts)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    gost.kernel(dm)

    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'  nao = {nao}, natm = {natm}, ngrids = {gost.n_gaussian}')
    print(f'  min area = {gost.areas.min():.3e}')
    print(f'  min F_g = {gost.forces.min():.3e}')
    print(f'{"="*60}')

    print('  Computing analytical dV/dR...')
    dV_ana = analytical_grad_vmat(gost, dm)

    print('  Computing numerical dV/dR (fdiff of kernel)...')
    dV_fd = numerical_grad_vmat(gost, dm)

    err = np.max(np.abs(dV_ana - dV_fd))
    ref = np.max(np.abs(dV_fd))
    rel = err / ref if ref > 1e-15 else 0.0

    print(f'\n  max |dV_fd|:  {ref:.3e}')
    print(f'  max |error|:  {err:.3e}')
    print(f'  relative:     {rel:.2e}')

    if err < 1e-7:
        print('  PASS ✓')
    elif rel < 1e-5:
        print(f'  PASS ✓ (abs err {err:.1e} but rel err OK)')
    else:
        print('  FAIL ✗')
        # Per-atom breakdown
        for A in range(natm):
            blk_err = np.max(np.abs(dV_ana[A] - dV_fd[A]))
            if blk_err > 1e-7:
                print(f'    atom {A}: {blk_err:.3e}')

    return dV_ana, dV_fd


if __name__ == '__main__':
    opts_vdw = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}

    test_grad_vmat('N 0 0 0; N 0 0 1.098', 'cc-pVDZ', opts_vdw,
                   'N2/cc-pVDZ/vdw')

    sf6_atom = '''S  0.000  0.000  0.000
                  F  1.560  0.000  0.000
                  F -1.560  0.000  0.000
                  F  0.000  1.560  0.000
                  F  0.000 -1.560  0.000
                  F  0.000  0.000  1.560
                  F  0.000  0.000 -1.560'''
    test_grad_vmat(sf6_atom, 'cc-pVDZ', opts_vdw, 'SF6/cc-pVDZ/vdw')
