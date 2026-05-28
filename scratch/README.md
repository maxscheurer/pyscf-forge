# Analytical Hessian — Term-by-Term Development

Scratch scripts for developing the explicit GOSTSHYP Hessian `d²E/dR²` at fixed density.
Each script tests one piece against finite differences before assembly.

## Energy Expression

```
E = P * Σ_g  A_g * e_g / F_g
```

where (at fixed D):
- `P` = pressure (constant)
- `A_g(R)` = area of grid point g (depends on geometry via tessellation)
- `e_g(R) = Tr[D · Gtilde_g(R)]` = s-type 3c overlap trace
- `F_g(R) = Tr[D · Fhat_g(R)]` = p-type 3c overlap contracted with normals, traced with D
- `ω_g(R) = π·ln2 / A_g(R)` = Gaussian exponent (depends on R through A)

## Gradient (first derivative)

```
dE/dR_Ax = P * Σ_g [ dA/dR_Ax · e/F  +  A/F · de/dR_Ax  -  A·e/F² · dF/dR_Ax ]
```

## Hessian (second derivative)

Differentiating the gradient w.r.t. R_By gives 10 terms:

```
d²E/(dR_Ax dR_By) = P * Σ_g [
  H1:  d²A/(dR_Ax dR_By) · e/F
  H2:  dA/dR_Ax · de/dR_By / F
  H3: -dA/dR_Ax · e · dF/dR_By / F²
  H4:  dA/dR_By · de/dR_Ax / F
  H5:  A/F · d²e/(dR_Ax dR_By)
  H6: -A/F² · de/dR_Ax · dF/dR_By
  H7: -dA/dR_By · e · dF/dR_Ax / F²
  H8: -A/F² · de/dR_By · dF/dR_Ax
  H9: -A·e/F² · d²F/(dR_Ax dR_By)
  H10: 2·A·e/F³ · dF/dR_Ax · dF/dR_By
]
```

## Ingredients Needed

### Already available:
- `e_g`, `F_g`, `A_g` — from `gost.kernel(dm)`
- `dA_g/dR` — from `get_dF_dA(surface)`, shape `(ngrids, natm, 3)`
- `de_g/dR`, `dF_g/dR` — scalar traces computed in `analytical_grad_vmat` as `dg_trace`, `dF_trace`

### New for Hessian:
- `d²A_g/(dR dR)` — from `get_d2F_d2A(surface)`, shape `(natm, natm, 3, 3, ngrids)`
- `d²e_g/(dR dR)` — second derivative of Gtilde trace (needs 2nd-deriv integrals)
- `d²F_g/(dR dR)` — second derivative of Fhat trace (needs 2nd-deriv integrals)

## Structure of d²e_g/(dR_Ax dR_By)

The Gtilde integral `<χ_i|exp(-ω|r-R_g|²)|χ_j>` depends on R through:
1. **Bra center R_i** (moves with atom of χ_i)
2. **Ket center R_j** (moves with atom of χ_j)
3. **Aux center R_g** (= grid point, moves with its atom)
4. **Exponent ω_g** (depends on all atoms through area A_g)

The first derivative `de_g/dR_Ax` collects:
- **bra+ket on A**: uses `int3c1e_ip1` (exploits integral symmetry in bra/ket)
- **aux on A**: uses `int3c1e_ip1` with aux shell slices (only if g is on atom A)
- **width on A**: uses d-type integrals via `dω/dR = -(ω/A)·dA/dR`

The second derivative `d²e_g/(dR_Ax dR_By)` has position×position, position×width, and width×width blocks:

### Position × Position (using 2nd-derivative integrals):

| Sub-term | When | Integral |
|----------|------|----------|
| ∂²/∂R_bra² (same atom A=B) | i∈A, differentiating R_A again | `int3c1e_ipip1` |
| ∂²/(∂R_bra ∂R_ket) | i∈A, j∈B | `int3c1e_ipvip1` |
| ∂²/(∂R_bra ∂R_aux) | i∈A, g on B | `int3c1e_ip1ip2` |
| ∂²/∂R_aux² (same atom) | g on A=B | `int3c1e_ipip2` |
| ∂²/(∂R_aux ∂R_ket) | g on A, j∈B | = -(ip1ip2 + ipip2) via transl. inv. |

Translational invariance: `ipip1 + ipvip1 + ip1ip2 = 0` (sum of all 2nd derivs w.r.t. bra = 0)

### Position × Width:
- d/dR_By of [dω/dR_Ax · Tr[D · ∂Gtilde/∂ω]]
- Requires ip1 derivatives of d-type integrals, plus d²ω/(dR dR) terms

### Width × Width:
- d/dR_By of [dω/dR_Ax · Tr[D · ∂Gtilde/∂ω]] when B also contributes through ω
- Requires g-type (l=4) integrals for ∂²Gtilde/∂ω²

## d²F_g/(dR_Ax dR_By) — Same Structure with p-type

Identical structure to d²e but:
- s-type fakemol → p-type fakemol (contracted with normals)
- d-type width correction → f-type
- Normal vectors are FIXED for vdw cavity (normals = Lebedev directions, geometry-independent)

## Testing Strategy (this directory)

Each script tests one ingredient against finite differences:
1. `common.py` — shared molecule/SCF/GOSTSHYP setup
2. `01_scalar_traces.py` — validate de_g/dR, dF_g/dR vs fdiff of e_g, F_g
3. `02_d2A.py` — validate d²A from get_d2F_d2A vs fdiff of dA
4. `03_d2e.py` — validate d²e_g/(dR dR) vs fdiff of de_g/dR
5. `04_d2F.py` — validate d²F_g/(dR dR) vs fdiff of dF_g/dR
6. `05_hess_assembly.py` — assemble all terms, compare to hess_fd

## Available 2nd-Derivative Integrals

All confirmed working in standard pyscf 2.13.0 libcint:

| Integral | Shape (H2/sto3g, 1 aux) | Meaning |
|----------|------------------------|---------|
| `int3c1e_ipip1` | (9, nao, nao, naux) | ∂²/∂R_bra² |
| `int3c1e_ipvip1` | (9, nao, nao, naux) | ∂²/(∂R_bra ∂R_ket) |
| `int3c1e_ip1ip2` | (9, nao, nao, naux) | ∂²/(∂R_bra ∂R_aux) |
| `int3c1e_ipip2` | (9, nao, nao, naux) | ∂²/∂R_aux² |

9 components = 3×3 Cartesian (xx, xy, xz, yx, yy, yz, zx, zy, zz).
All work with l=0,1,2,3 fakemols.
