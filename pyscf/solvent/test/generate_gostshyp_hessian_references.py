#!/usr/bin/env python
"""
Generate reference Hessians for GOSTSHYP end-to-end tests.

Computes numerical Hessians by central finite differences of the full
SCF+GOSTSHYP gradient at displaced geometries. Results are stored as
.npy files in the reference_data/ subdirectory.

Usage:
    python -m pyscf.solvent.test.generate_gostshyp_hessian_references [--systems sys1 sys2 ...]
    python -m pyscf.solvent.test.generate_gostshyp_hessian_references --all

Without arguments, prints available systems.
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP

from pyscf.solvent.test.reference_systems import SYSTEMS


REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference_data')


def numerical_hessian(mol, gost_opts, step=1e-4):
    """Full numerical Hessian via central fdiff of SCF+GOSTSHYP gradient."""
    natm = mol.natm
    coords0 = mol.atom_coords().copy()
    hess = np.zeros((natm, 3, natm, 3))

    for B in tqdm(range(natm), desc='    fdiff', leave=False):
        for y in range(3):
            for sign, s in [(+1, step), (-1, -step)]:
                coords = coords0.copy()
                coords[B, y] += s
                mol_d = mol.copy()
                mol_d.set_geom_(coords, unit='Bohr')
                gost_d = GOSTSHYP(mol_d, options=gost_opts)
                mf_d = scf.RHF(mol_d).GOSTSHYP(solvent_obj=gost_d)
                mf_d.conv_tol = 1e-12
                mf_d.conv_tol_grad = 1e-10
                mf_d.verbose = 0
                mf_d.kernel()
                assert mf_d.converged, (
                    f'SCF did not converge for atom {B} coord {y} '
                    f'sign {sign}')
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


def generate(system_name):
    """Generate and save a reference Hessian for one system."""
    system = SYSTEMS[system_name]
    mol = gto.M(atom=system.atom, basis=system.basis,
                unit=system.unit, verbose=0)
    gost_opts = system.gostshyp.to_dict()

    print(f'  {system_name}: {mol.natm} atoms, basis={system.basis}')

    # Also compute analytical for comparison
    mf = system.make_mf()
    gost = mf.with_solvent
    print(f'    E = {mf.e_tot:.10f}, ngrids = {gost.n_gaussian}, '
          f'min_area = {gost.areas.min():.3e}')

    t0 = time.time()
    hess_num = numerical_hessian(mol, gost_opts)
    dt = time.time() - t0
    print(f'    numerical Hessian computed in {dt:.1f}s')

    hess_ana = mf.Hessian().kernel()
    err = np.max(np.abs(hess_ana - hess_num))
    ref = np.max(np.abs(hess_num))
    print(f'    max |H_num| = {ref:.3e}, max |error| = {err:.3e}, '
          f'rel = {err/ref:.2e}')

    path = os.path.join(REFERENCE_DIR, f'hess_{system_name}.npy')
    np.save(path, hess_num)
    print(f'    saved to {path}')
    return hess_num


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems', nargs='+', default=None,
                        help='Systems to generate (default: list available)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all systems')
    args = parser.parse_args()

    os.makedirs(REFERENCE_DIR, exist_ok=True)

    if args.all:
        names = list(SYSTEMS.keys())
    elif args.systems:
        names = args.systems
        for n in names:
            if n not in SYSTEMS:
                print(f'Unknown system: {n}')
                print(f'Available: {list(SYSTEMS.keys())}')
                sys.exit(1)
    else:
        print('Available systems:')
        for name, system in SYSTEMS.items():
            path = os.path.join(REFERENCE_DIR, f'hess_{name}.npy')
            exists = '✓' if os.path.exists(path) else '✗'
            print(f'  [{exists}] {name} ({system.molecule}/{system.basis})')
        print(f'\nUse --systems name1 name2 or --all')
        return

    print(f'Generating {len(names)} reference Hessian(s)...\n')
    for name in names:
        generate(name)
        print()


if __name__ == '__main__':
    main()
