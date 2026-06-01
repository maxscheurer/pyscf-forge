#!/usr/bin/env python
"""
Test 12: Validate GOSTSHYP._B_dot_x against finite differences.

For a random symmetric dm1, compares _B_dot_x(dm1) against
(V(D + ε·dm1) - V(D - ε·dm1)) / (2ε), where V is the full GOSTSHYP
Fock matrix computed by kernel().

Tests on:
  - H2/sto-3g (minimal, fast)
  - N2/cc-pVDZ (realistic)

Also tests both direct and cached modes.
"""
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP


def make_random_symmetric_dm(nao, seed=42):
    """Generate a random symmetric matrix to use as dm1."""
    rng = np.random.default_rng(seed)
    dm1 = rng.standard_normal((nao, nao))
    dm1 = 0.5 * (dm1 + dm1.T)
    return dm1


def numerical_B_dot_x(gost, dm, dm1, eps=1e-5):
    """Compute _B_dot_x via central finite differences of kernel().

    Returns (V(D + ε·dm1) - V(D - ε·dm1)) / (2ε).
    """
    # V(D + ε·dm1)
    gost_p = GOSTSHYP(gost.mol, options=get_opts(gost))
    gost_p.kernel(dm + eps * dm1)
    v_plus = gost_p.v.copy()

    # V(D - ε·dm1)
    gost_m = GOSTSHYP(gost.mol, options=get_opts(gost))
    gost_m.kernel(dm - eps * dm1)
    v_minus = gost_m.v.copy()

    return (v_plus - v_minus) / (2.0 * eps)


def get_opts(gost):
    """Extract options dict from a GOSTSHYP object."""
    opts = {
        'cavity': gost.cavity,
        'pressure_mpa': gost.pressure_mpa,
        'npoints': gost.npoints,
        'scaling_factor': gost.scaling_factor,
        'direct': gost.direct,
    }
    if gost.cavity == 'vdw/occ':
        opts['r_ext'] = gost.r_ext
    return opts


def test_B_dot_x(atom, basis, gost_opts, label, direct=True):
    """Run _B_dot_x test for a given molecule and mode."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    nao = mol.nao_nr()

    gost_opts = dict(gost_opts, direct=direct)
    gost = GOSTSHYP(mol, options=gost_opts)

    # Converge SCF to get a proper dm
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    # Run kernel to populate amplitudes/forces/gtilde_expval
    gost.kernel(dm)

    dm1 = make_random_symmetric_dm(nao)

    # Analytical
    v_ana = gost._B_dot_x(dm1)

    # Numerical
    v_num = numerical_B_dot_x(gost, dm, dm1, eps=1e-5)

    err = np.max(np.abs(v_ana - v_num))
    ref = np.max(np.abs(v_num))
    rel = err / ref if ref > 1e-15 else 0.0

    mode_str = 'direct' if direct else 'cached'
    print(f'\n{"="*60}')
    print(f'{label} ({mode_str})')
    print(f'  nao = {nao}, ngrids = {gost.n_gaussian}')
    print(f'  max |B·x_num|: {ref:.3e}')
    print(f'  max |error|:   {err:.3e}')
    print(f'  relative:      {rel:.2e}')

    if err < 1e-8:
        print('  PASS ✓')
    elif rel < 1e-6:
        print(f'  PASS ✓ (abs err {err:.1e} but rel err OK)')
    else:
        print('  FAIL ✗')

        # Print per-element detail for debugging
        idx = np.unravel_index(np.argmax(np.abs(v_ana - v_num)), v_ana.shape)
        print(f'    worst element at {idx}: ana={v_ana[idx]:.10e} num={v_num[idx]:.10e}')

    return v_ana, v_num


def test_batched(atom, basis, gost_opts, label):
    """Test that batched dm1 (nset, nao, nao) works correctly."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    nao = mol.nao_nr()

    gost = GOSTSHYP(mol, options=gost_opts)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()
    gost.kernel(dm)

    rng = np.random.default_rng(123)
    nset = 3
    dm1_batch = rng.standard_normal((nset, nao, nao))
    dm1_batch = 0.5 * (dm1_batch + dm1_batch.transpose(0, 2, 1))

    # Batched call
    v_batch = gost._B_dot_x(dm1_batch)

    # Single calls
    v_singles = np.array([gost._B_dot_x(dm1_batch[i]) for i in range(nset)])

    err = np.max(np.abs(v_batch - v_singles))
    print(f'\n{"="*60}')
    print(f'{label} — batched vs single')
    print(f'  max |error|: {err:.3e}')
    if err < 1e-14:
        print('  PASS ✓')
    else:
        print('  FAIL ✗')


if __name__ == '__main__':
    opts_vdw = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}

    # --- H2/sto-3g ---
    test_B_dot_x('H 0 0 0; H 0 0 0.74', 'sto-3g', opts_vdw,
                 'H2/sto-3g/vdw', direct=True)
    test_B_dot_x('H 0 0 0; H 0 0 0.74', 'sto-3g', opts_vdw,
                 'H2/sto-3g/vdw', direct=False)

    # --- N2/cc-pVDZ ---
    test_B_dot_x('N 0 0 0; N 0 0 1.098', 'cc-pVDZ', opts_vdw,
                 'N2/cc-pVDZ/vdw', direct=True)
    test_B_dot_x('N 0 0 0; N 0 0 1.098', 'cc-pVDZ', opts_vdw,
                 'N2/cc-pVDZ/vdw', direct=False)

    # --- SF6/cc-pVDZ ---
    sf6_atom = '''S  0.000  0.000  0.000
                  F  1.560  0.000  0.000
                  F -1.560  0.000  0.000
                  F  0.000  1.560  0.000
                  F  0.000 -1.560  0.000
                  F  0.000  0.000  1.560
                  F  0.000  0.000 -1.560'''
    test_B_dot_x(sf6_atom, 'cc-pVDZ', opts_vdw,
                 'SF6/cc-pVDZ/vdw', direct=True)

    # --- Batched test ---
    test_batched('H 0 0 0; H 0 0 0.74', 'sto-3g', opts_vdw,
                 'H2/sto-3g/vdw')
