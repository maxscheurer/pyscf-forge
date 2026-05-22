#!/usr/bin/env python

'''
Geometry optimization of HF under hydrostatic pressure using the GOSTSHYP model.

The GOSTSHYP (Gaussian On Surface Tesserae Simulate HYdrostatic Pressure)
model applies isotropic pressure to a molecular cavity, compressing the
molecule.  Here we optimize the H-F bond length at 50 GPa and compare to
the gas-phase equilibrium geometry.

Reference:
    J. Chem. Theory Comput. 2021, 17, 1, 583-597
    https://doi.org/10.1021/acs.jctc.0c01212
'''

from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, gostshyp_for_scf

# Build HF molecule with an initial bond length of 0.95 Angstrom
mol = gto.M(
    atom='H 0 0 0; F 0 0 0.95',
    basis='cc-pVDZ',
    verbose=4,
)

#
# 1. Gas-phase geometry optimization (reference)
#
mf_gas = scf.RHF(mol).run()
grad_gas = mf_gas.Gradients().as_scanner()
mol_gas = grad_gas.optimizer().kernel()
print(f'\nGas-phase equilibrium bond length: '
      f'{mol_gas.atom_coord(1)[2] - mol_gas.atom_coord(0)[2]:.4f} Bohr')

#
# 2. Geometry optimization under 50 GPa pressure
#
mf = scf.RHF(mol)
gost = GOSTSHYP(mol, options={'pressure_mpa': 50_000, 'cavity': 'vdw/occ'})
mf = gostshyp_for_scf(mf, gost)
mf.kernel()

grad = mf.Gradients().as_scanner()
mol_pressed = grad.optimizer().kernel()
print(f'\nCompressed equilibrium bond length (50 GPa): '
      f'{mol_pressed.atom_coord(1)[2] - mol_pressed.atom_coord(0)[2]:.4f} Bohr')

#
# 3. Summary
#
r_gas = mol_gas.atom_coord(1)[2] - mol_gas.atom_coord(0)[2]
r_pressed = mol_pressed.atom_coord(1)[2] - mol_pressed.atom_coord(0)[2]
print(f'\nBond compression: {(r_gas - r_pressed):.4f} Bohr '
      f'({(r_gas - r_pressed) * 0.529177:.4f} Angstrom)')
