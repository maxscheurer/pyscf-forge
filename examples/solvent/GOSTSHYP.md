# GOSTSHYP: Hydrostatic Pressure in PySCF

GOSTSHYP (Gaussian On Surface Tesserae Simulate HYdrostatic Pressure) applies
isotropic pressure to a molecular cavity surface, enabling high-pressure
electronic-structure calculations in PySCF.

## Installation

Install this pyscf-forge branch on top of PySCF:

```bash
pip install git+https://github.com/maxscheurer/pyscf-forge.git@gostshyp
```

The standard installation does not install JAX, CavJAX, or MOIST. CavJAX is
temporarily pinned to its reviewed public commit and can be installed with:

```bash
pip install "cavjax @ git+https://github.com/maxscheurer/cavjax.git@b2d0aa8"
# When the package-index release is available, use:
# pip install "pyscf-forge[cavjax]"
```

CavJAX requires Python 3.10 or newer and JAX float64. Set `JAX_ENABLE_X64=1`
before Python starts. The GOSTSHYP bridge executes CavJAX construction, surface
builds, and responses on a JAX CPU device without changing process-global JAX
platform selection.

## Quick start

```python
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, gostshyp_for_scf

mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='cc-pVDZ')
gost = GOSTSHYP(mol, options={'pressure_mpa': 50_000})
mf = gostshyp_for_scf(scf.RHF(mol), gost)
mf.kernel()
```

Importing or using ordinary GOSTSHYP cavities does not import or require
CavJAX. After importing `pyscf.solvent.gostshyp`, method syntax is also
available:

```python
mf = scf.RHF(mol).GOSTSHYP()
mf.kernel()
```

## Options

| Parameter | Default | Description |
|---|---:|---|
| `pressure_mpa` | 50000 | Applied pressure in MPa |
| `cavity` | `'vdw/occ'` | `'vdw'`, `'vdw/occ'`, `'drop'`, or `'cavjax'` |
| `npoints` | 110 | Lebedev order for atom-centered `vdw`, `vdw/occ`, and `drop` surfaces |
| `scaling_factor` | 1.2 | Radius scale for legacy atom-centered surfaces |
| `r_ext` | 0.4724 | `vdw/occ` extension radius in Bohr |
| `direct` | `True` | Evaluate electronic integrals in bounded chunks |
| `drop_kwargs` | `None` | Options forwarded to MOIST DROPSvdW |
| `cavjax_kwargs` | `{}` | Options forwarded unchanged to `MolecularCavity` |

CavJAX is direct-only; `cavity='cavjax', direct=False` is rejected during
construction before cached integral tensors can be created.

## CavJAX example

```python
# Start Python with JAX_ENABLE_X64=1.
gost = GOSTSHYP(mol, options={
    'cavity': 'cavjax',
    'direct': True,
    'pressure_mpa': 50_000,
    'cavjax_kwargs': {
        'points_per_atom': 100,
        'radius_scale': 1.0,
        'radius_offset': 0.0,             # Bohr
        'target_shape_directions': 162,
        'support_temperature': 0.2834589187,  # Bohr
        'wall_temperature': 0.1511780900,     # Bohr
        'root_steps': 48,
    },
})
```

`points_per_atom` specifies an approximate **global** point budget
`natm * points_per_atom`; it does not create an atom-centered grid. CavJAX
rounds this target and `target_shape_directions` to supported counts of the
form `10*f**2 + 2`. The resolved counts are logged and available from the
retained backend.

All coordinates and length options passed across the integration boundary use
Bohr, and areas use Bohr². CavJAX produces outward normals. GOSTSHYP stores
inward normals, converts them once at surface construction, and converts the
normal cotangent once before the CavJAX response.

A `GOSTSHYP` object retains one fixed-topology `MolecularCavity`. `reset()` with
new coordinates and unchanged atomic numbers/order reuses that object and
rebuilds the geometry. A changed composition or atom order is rejected; create
a new `GOSTSHYP` object instead.

The direct gradient keeps only point `(Q,3)`, area `(Q,)`, and normal `(Q,3)`
cotangents, then makes one contracted CavJAX response call. It does not form a
dense cavity Jacobian or a full-grid `nao² * Q * 3` normal tensor.

Last-call timings are available as:

- `_cavjax_build_t_wall`: CPU construction/build and host transfer;
- `_grad_integral_t_wall`: direct-gradient libcint calls;
- `_grad_cotangent_t_wall`: NumPy cotangent contraction and assembly;
- `_grad_cavjax_vjp_t_wall`: contracted CPU response and host transfer.

`_t_wall` remains accumulated GOSTSHYP kernel time, while `_grad_t_wall` is the
last overall gradient time.

## Analytic gradients and geometry optimization

```python
mf = gostshyp_for_scf(scf.RHF(mol), gost)
mf.kernel()
grad = mf.Gradients().kernel()
scanner = mf.Gradients().as_scanner()
mol_opt = scanner.optimizer().kernel()
```

See `examples/solvent/00-gostshyp_geomopt.py` for a complete legacy-cavity
example.

## Cavity types

- **`vdw`** — scaled Bondi spheres with atom-centered Lebedev points; points
  inside neighboring spheres are removed.
- **`vdw/occ`** — the van der Waals surface with an outer-cavity correction that
  removes points in crevices.
- **`drop`** — optional MOIST DROPSvdW cavity.
- **`cavjax`** — one smooth, global, fixed-topology convex envelope with a
  contracted differentiable geometry response.

## Limitations

- GOSTSHYP wraps SCF methods only (RHF, UHF, RKS, and UKS), not post-SCF methods.
- The pressure model assumes a suitable closed cavity; convergence and an
  internally consistent gradient do not establish physical validity.
- CavJAX's current envelope is strictly convex. It cannot represent arbitrary
  concave molecular surfaces or disconnected cavities.
- Finite shape directions introduce orientation dependence. Increasing
  `target_shape_directions` reduces but does not eliminate this discretization
  effect.
- The smoothing temperatures inflate the envelope, and the positive
  vertex-centered quadrature is not equal-area. Point counts therefore do not
  map one-to-one to Lebedev orders or physical accuracy.
- CavJAX integration correctness is not universal validation of its cavity for
  every molecule, cluster, pressure, or chemical application.

## References

1. Scheurer, M.; Dreuw, A.; Epifanovsky, E.; Head-Gordon, M.; Stauch, T.
   *J. Chem. Theory Comput.* **2021**, 17, 583–597.
   [doi:10.1021/acs.jctc.0c01212](https://doi.org/10.1021/acs.jctc.0c01212)
2. Pausch, A.; Zeller, F.; Neudecker, T.
   *J. Chem. Theory Comput.* **2025**, 21, 747–761.
   [doi:10.1021/acs.jctc.4c01502](https://doi.org/10.1021/acs.jctc.4c01502)
