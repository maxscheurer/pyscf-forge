# GOSTSHYP: Hydrostatic Pressure in PySCF

GOSTSHYP (Gaussian On Surface Tesserae Simulate HYdrostatic Pressure) applies
isotropic pressure to a molecular cavity surface, enabling high-pressure
electronic structure calculations within PySCF.

## Installation

GOSTSHYP is on the `gostshyp` branch of the pyscf-forge fork. Install it on
top of an existing PySCF installation:

```bash
pip install git+https://github.com/maxscheurer/pyscf-forge.git@gostshyp
```

Or clone and install in editable mode:

```bash
git clone -b gostshyp https://github.com/maxscheurer/pyscf-forge.git
pip install --no-deps -e pyscf-forge
```

pyscf-forge extends PySCF via its plugin system — no patching required.

## Quick Start

```python
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, gostshyp_for_scf

mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='cc-pVDZ')

# Create a GOSTSHYP object (50 GPa pressure)
gost = GOSTSHYP(mol, options={'pressure_mpa': 50_000})

# Wrap an SCF object with the pressure model
mf = scf.RHF(mol)
mf = gostshyp_for_scf(mf, gost)
mf.kernel()
```

Alternatively, after importing the solvent module you can use the method syntax:

```python
from pyscf.solvent import gostshyp  # registers .GOSTSHYP() on SCF objects

mf = scf.RHF(mol).GOSTSHYP()
mf.kernel()
```

## Options

| Parameter        | Default    | Description                                      |
|------------------|------------|--------------------------------------------------|
| `pressure_mpa`   | 50000      | Applied pressure in MPa (50000 MPa = 50 GPa)   |
| `cavity`         | `'vdw/occ'`| Cavity type: `'vdw'` or `'vdw/occ'` (occluded) |
| `npoints`        | 110        | Lebedev grid points per atom                     |
| `scaling_factor` | 1.2        | Van der Waals radii scaling factor               |
| `r_ext`          | 0.4724     | Extension radius for vdW/OCC cavity (Bohr)       |

Pass options as a dictionary:

```python
gost = GOSTSHYP(mol, options={
    'pressure_mpa': 100_000,
    'cavity': 'vdw',
    'npoints': 302,
})
```

## Analytic Gradients and Geometry Optimization

GOSTSHYP supports analytic nuclear gradients, enabling geometry optimization
under pressure:

```python
mf = gostshyp_for_scf(scf.RHF(mol), gost)
mf.kernel()

# Analytic gradient
grad = mf.Gradients().kernel()

# Geometry optimization (requires geomeTRIC)
scanner = mf.Gradients().as_scanner()
mol_opt = scanner.optimizer().kernel()
```

See `examples/solvent/00-gostshyp_geomopt.py` for a complete working example
that compresses H-F under 50 GPa.

## Cavity Types

- **`'vdw'`** — Standard van der Waals surface built from scaled Bondi radii
  with a Lebedev angular grid on each atom. Grid points inside neighboring
  spheres are removed.

- **`'vdw/occ'`** — Van der Waals surface with Outer Cavity Correction. An
  extended probe sphere removes grid points in crevices between atoms, giving a
  smoother cavity that avoids unphysical pressure concentration in bonding
  regions.

## Limitations

- GOSTSHYP wraps **SCF methods only** (RHF, UHF, RKS, UKS). Post-SCF methods
  (MP2, CCSD, etc.) are not supported and will raise `TypeError`.
- The model assumes a closed, convex-like cavity. Very extended or fragmented
  molecules may give unphysical results.

## References

1. Lorenz, S.; Raich, F.; Eichkorn, K.; Apostolidis, C.; Kästner, J.
   *J. Chem. Theory Comput.* **2021**, 17, 583–597.
   [doi:10.1021/acs.jctc.0c01212](https://doi.org/10.1021/acs.jctc.0c01212)

2. Lorenz, S.; Kästner, J.
   *J. Chem. Theory Comput.* **2025**, 21, 764–776.
   [doi:10.1021/acs.jctc.4c01502](https://doi.org/10.1021/acs.jctc.4c01502)
