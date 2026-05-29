#!/usr/bin/env python
"""
Test 11: Understand PCM Hessian structure as reference for GOSTSHYP.

Questions:
  1. What does pcm.hess(dm) compute? (direct term at fixed D)
  2. Does PCM make_h1 == fdiff of the PCM Fock operator?
  3. Does the full Hessian (mf.Hessian().kernel()) agree with fdiff of gradient?

Use H2/sto-3g for speed.
"""
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.pcm import PCM

# Use H2 for speed (same as GOSTSHYP tests)
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', unit='Angstrom', verbose=0)

# --- PCM SCF ---
mf = scf.RHF(mol).PCM()
mf.with_solvent.method = 'C-PCM'
mf.conv_tol = 1e-12
mf.kernel()
dm = mf.make_rdm1()
pcm = mf.with_solvent

print(f'H2/sto-3g C-PCM: E = {mf.e_tot:.10f}')
print(f'  nao={mol.nao_nr()}, natm={mol.natm}')

# --- 1. Direct term: pcm.hess(dm) ---
hess_direct = pcm.hess(dm)
print(f'\n1. pcm.hess(dm):')
print(f'   shape = {hess_direct.shape}')
print(f'   max = {np.max(np.abs(hess_direct)):.6e}')

# --- 2. PCM make_h1: analytical vs fdiff of PCM Fock contribution ---
hessobj = mf.Hessian()
mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
h1_ana = hessobj.make_h1(mo_coeff, mo_occ)  # includes vacuum + PCM

# Also get the vacuum-only make_h1 for comparison
mf_vac = scf.RHF(mol)
mf_vac.conv_tol = 1e-12
mf_vac.kernel()
hessobj_vac = mf_vac.Hessian()
h1_vac = hessobj_vac.make_h1(mf_vac.mo_coeff, mf_vac.mo_occ)

# The PCM contribution to make_h1 is h1_ana - h1_vac (approximately;
# MO coefficients differ, but for fixed-D comparison this is informative)
# Better: use the analytical_grad_vmat directly
from pyscf.solvent.hessian.pcm import analytical_grad_vmat as pcm_grad_vmat
dV_pcm_ana = pcm_grad_vmat(pcm, dm)

# Numerical: fdiff of the PCM Fock matrix V_PCM(D) at fixed D
step = 1e-5
natm = mol.natm
nao = mol.nao_nr()
coords0 = mol.atom_coords().copy()

dV_pcm_fd = np.zeros((natm, 3, nao, nao))
for atm in range(natm):
    for x in range(3):
        coords_p = coords0.copy(); coords_p[atm, x] += step
        mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr')
        pcm_p = PCM(mol_p); pcm_p.method = 'C-PCM'
        _, v_p = pcm_p.kernel(dm)

        coords_m = coords0.copy(); coords_m[atm, x] -= step
        mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr')
        pcm_m = PCM(mol_m); pcm_m.method = 'C-PCM'
        _, v_m = pcm_m.kernel(dm)

        dV_pcm_fd[atm, x] = (v_p - v_m) / (2 * step)

err_h1 = np.max(np.abs(dV_pcm_ana - dV_pcm_fd))
ref_h1 = np.max(np.abs(dV_pcm_fd))
print(f'\n2. PCM make_h1 (analytical_grad_vmat) vs fdiff of V_PCM:')
print(f'   max |dV_fd| = {ref_h1:.6e}')
print(f'   max error   = {err_h1:.6e}')
print(f'   relative    = {err_h1/ref_h1:.2e}')

# --- 3. Full Hessian ---
hess_full = hessobj.kernel()
print(f'\n3. mf.Hessian().kernel():')
print(f'   de_solute  max = {np.max(np.abs(hessobj.de_solute)):.6e}')
print(f'   de_solvent max = {np.max(np.abs(hessobj.de_solvent)):.6e}')
print(f'   de_total   max = {np.max(np.abs(hess_full)):.6e}')

# Numerical Hessian: fdiff of full SCF+PCM gradient
hess_num = np.zeros((natm, 3, natm, 3))
step = 1e-4
for B in range(natm):
    for y in range(3):
        coords_p = coords0.copy(); coords_p[B, y] += step
        mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr')
        mf_p = scf.RHF(mol_p).PCM()
        mf_p.with_solvent.method = 'C-PCM'
        mf_p.conv_tol = 1e-12; mf_p.verbose = 0; mf_p.kernel()
        gp = mf_p.nuc_grad_method(); gp.verbose = 0; grad_p = gp.kernel()

        coords_m = coords0.copy(); coords_m[B, y] -= step
        mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr')
        mf_m = scf.RHF(mol_m).PCM()
        mf_m.with_solvent.method = 'C-PCM'
        mf_m.conv_tol = 1e-12; mf_m.verbose = 0; mf_m.kernel()
        gm = mf_m.nuc_grad_method(); gm.verbose = 0; grad_m = gm.kernel()

        hess_num[:, :, B, y] = (grad_p - grad_m) / (2 * step)

hess_num = hess_num.transpose(0, 2, 1, 3)
hess_num = 0.5 * (hess_num + hess_num.transpose(1, 0, 3, 2))

err_full = np.max(np.abs(hess_full - hess_num))
ref_full = np.max(np.abs(hess_num))
print(f'\n   Full analytical vs fdiff-of-gradient:')
print(f'   max |H_num| = {ref_full:.6e}')
print(f'   max error   = {err_full:.6e}')
print(f'   relative    = {err_full/ref_full:.2e}')
