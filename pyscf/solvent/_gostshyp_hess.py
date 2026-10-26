# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Analytical Hessian for the GOSTSHYP pressure model.

Computes d²E_GOSTSHYP/(dR_A dR_B) at fixed density matrix (the direct term).
The GOSTSHYP energy per grid point is:  E_g = P · A_g · e_g / F_g,
so the Hessian involves derivatives of areas (A), the Gtilde trace (e),
and the force trace (F) up to second order.

The assembly uses 10 terms (H1–H10) from the product rule applied to A·e/F.
Each of d²e and d²F decomposes into 5 groups:
  G1: position × position
  G2: forward cross (dω/dR_By · ∂²/∂pos_Ax∂ω)
  G3: reverse cross (dω/dR_Ax · ∂²/∂pos_By∂ω)
  G4: d²ω piece (d²ω/(dR_Ax dR_By) · ∂/∂ω)
  G5: ω×ω piece (dω/dR_Ax · dω/dR_By · ∂²/∂ω²)

Implementation notes:
  - Every auxiliary function depends on r - K only, so ∂/∂K = -(∂/∂R_i + ∂/∂R_j).
    Position derivatives therefore need only bra-side integrals (ip1, ipip1,
    ipvip1); the Gaussian-center contributions follow from a translation
    projection (_project1, _project2).
  - Width derivatives multiply the Gaussian by powers of r_K², which are
    evaluated with int3c1e_r{2,4}_origk on s- or p-type auxiliaries.
  - D is symmetrized and contracted directly with the integral buffers;
    grid points are processed in chunks bounded by gost.max_memory.
"""

import numpy as np

from pyscf import lib
from pyscf.solvent.gostshyp import fakemol_for_gaussian
from pyscf.solvent.grad.pcm import get_dF_dA
from pyscf.solvent.hessian.pcm import get_d2F_d2A


def kernel(gost, dm):
    """Compute the full analytical GOSTSHYP Hessian contribution.

    Parameters
    ----------
    gost : GOSTSHYP object with kernel() already called
    dm : ndarray (nao, nao)

    Returns
    -------
    hess : ndarray (natm, natm, 3, 3)
    """
    mol = gost.mol
    natm = mol.natm
    P = gost.pressure_au
    e_g = gost.gtilde_expval
    F_g = gost.forces
    A_g = gost.areas

    # Inactive grid points: negative amplitudes clamped to zero in
    # gostshyp.kernel(), forces set to sentinel 1.0.  These must not
    # contribute to the Hessian.
    inactive = (gost.amplitudes == 0.0)

    dm = _symmetrize(dm)
    dA, d2A = _area_derivatives(gost)
    first = _first_order_terms(gost, dm)

    dg_trace, dF_trace = _compute_scalar_traces(gost, dm, first=first, dA=dA)
    d2e = _compute_d2e(gost, dm, first=first, dA=dA, d2A=d2A)
    d2F = _compute_d2F(gost, dm, first=first, dA=dA, d2A=d2A)

    # Assemble: E = P · Σ_g A·e/F → 10 terms from product rule
    inv_F = 1.0 / F_g
    inv_F2 = inv_F ** 2
    inv_F3 = inv_F ** 3

    # Zero out assembly prefactors for inactive grid points so they
    # contribute nothing to the Hessian sum.
    if inactive.any():
        e_g = e_g.copy()
        e_g[inactive] = 0.0
        inv_F[inactive] = 0.0
        inv_F2[inactive] = 0.0
        inv_F3[inactive] = 0.0

    hess = np.zeros((natm, natm, 3, 3))
    # H1: d²A · e/F
    hess += P * np.einsum('ABxyg,g->ABxy', d2A, e_g * inv_F)
    # H2 + H4: dA_Ax · de_By / F + dA_By · de_Ax / F
    hess += P * _outer_sum(dA, dg_trace * inv_F)
    hess += P * _outer_sum(dg_trace, dA * inv_F)
    # H3 + H7: -dA_Ax · e · dF_By / F² - dA_By · e · dF_Ax / F²
    hess -= P * _outer_sum(dA, dF_trace * (e_g * inv_F2))
    hess -= P * _outer_sum(dF_trace, dA * (e_g * inv_F2))
    # H5: A/F · d²e
    hess += P * np.einsum('ABxyg,g->ABxy', d2e, A_g * inv_F)
    # H6 + H8: -A/F² · (de_Ax · dF_By + de_By · dF_Ax)
    hess -= P * _outer_sum(dg_trace, dF_trace * (A_g * inv_F2))
    hess -= P * _outer_sum(dF_trace, dg_trace * (A_g * inv_F2))
    # H9: -A·e/F² · d²F
    hess -= P * np.einsum('ABxyg,g->ABxy', d2F, A_g * e_g * inv_F2)
    # H10: 2·A·e/F³ · dF_Ax · dF_By
    hess += 2.0 * P * _outer_sum(dF_trace, dF_trace * (A_g * e_g * inv_F3))

    hess = 0.5 * (hess + hess.transpose(1, 0, 3, 2))

    if not np.isfinite(hess).all():
        log = lib.logger.new_logger(gost)
        log.warn('GOSTSHYP Hessian contains non-finite values. '
                 'This typically occurs when area_thresh is disabled '
                 '(None/0) and the surface has very small-area grid points. '
                 'Consider setting area_thresh > 0.')

    return hess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetrize(dm):
    # All integral operators are symmetric in the AO pair, so only the
    # symmetric part of D contributes to any trace.
    dm = np.asarray(dm)
    return 0.5 * (dm + dm.T)


def _area_derivatives(gost):
    """First and second area derivatives, (natm,3,ng) and (natm,natm,3,3,ng)."""
    if gost._outer_surface_dict is not None:
        surface, scale = gost._outer_surface_dict, gost._occ_ratio_sq
    else:
        surface, scale = gost.surface_dict, 1.0
    _, dA = get_dF_dA(surface)
    _, d2A = get_d2F_d2A(surface)
    return dA.transpose(1, 2, 0) * scale, d2A * scale


def _grid_chunks(gost, n_nao2_per_grid):
    """Grid slices such that n_nao2_per_grid · nao² doubles per point fit in memory."""
    nao = gost.mol.nao_nr()
    max_memory = max(2000, gost.max_memory * 0.9 - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 1e6 / (8 * n_nao2_per_grid * nao**2)))
    ngrids = gost.n_gaussian
    for g0 in range(0, ngrids, blksize):
        yield g0, min(g0 + blksize, ngrids)


def _ao_to_atom(mol):
    """(natm, nao) indicator matrix: U[A, i] = 1 if AO i sits on atom A."""
    aoslice = mol.aoslice_by_atom()
    U = np.zeros((mol.natm, mol.nao_nr()))
    for A, (p0, p1) in enumerate(aoslice[:, 2:4]):
        U[A, p0:p1] = 1.0
    return U


def _grid_to_atom(gost):
    """(natm, ng) indicator matrix: O[A, g] = 1 if grid point g belongs to atom A."""
    O = np.zeros((gost.mol.natm, gost.n_gaussian))
    O[gost.atom_idx, np.arange(gost.n_gaussian)] = 1.0
    return O


def _int3c(mol, coords, widths, intor, comp, l=0, coeffs=None):
    """(comp, nao, nao, naux) integrals with auxiliary Gaussians on the grid."""
    gmol = fakemol_for_gaussian(coords, widths, l=l, coeffs=coeffs)
    supermol = mol + gmol
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)
    return supermol.intor(intor, comp=comp, shls_slice=shls_slice)


def _contract_dm(ints, dm):
    """Σ_j D_ij ints[..., i, j, a] → (..., nao, naux).  D must be symmetric.

    The integral buffers are Fortran-ordered in (i, j, a); writing D as D_ji
    keeps both operands unit-stride along i, so no copies are made.
    """
    return np.einsum('...ija,ji->...ia', ints, dm)


def _contract_normals(x, normals):
    """Contract the trailing p-type auxiliary component with the normals."""
    return np.einsum('...gc,gc->...g', x.reshape(x.shape[:-1] + (-1, 3)), normals)


def _trace(ints, dm, normals=None):
    """Tr[D · ints_a] for single-component integrals → (ng,)."""
    t = np.einsum('ija,ji->a', ints, dm)
    if normals is not None:
        t = _contract_normals(t, normals)
    return t


def _deriv1(ints, dm, U, O, normals=None):
    """Position derivative of Tr[D (ij|g)] at fixed width → (natm, 3, ng).

    ints are ip1 integrals (3, nao, nao, naux).  With symmetric D the bra and
    ket terms are equal, and the Gaussian-center term is fixed by translation
    invariance (_project1).
    """
    s = _contract_dm(ints, dm)
    if normals is not None:
        s = _contract_normals(s, normals)
    # ip1 = <∇i| = -∂/∂R_i; factor 2 for bra + ket
    t = -2.0 * np.tensordot(U, s, axes=([1], [1]))
    return _project1(t, O)


def _deriv2(mol, coords, widths, dm, U, O, l=0, coeffs=None, normals=None):
    """Position second derivative of Tr[D (ij|g)] at fixed width.

    Returns (natm, natm, 3, 3, ng).  Only ipip1 (∂_i∂_i) and ipvip1 (∂_i∂_j)
    are needed; Gaussian-center terms follow from _project2.
    """
    natm = mol.natm
    aoslice = mol.aoslice_by_atom()

    def reduce(ints, dm_part):
        s = _contract_dm(ints, dm_part)
        if normals is not None:
            s = _contract_normals(s, normals)
        return np.tensordot(U, s, axes=([1], [1]))   # (natm, 9, ng)

    ints = _int3c(mol, coords, widths, 'int3c1e_ipip1', 9, l, coeffs)
    ng = O.shape[1]
    M = np.zeros((natm, natm, 9, ng))
    M[np.arange(natm), np.arange(natm)] = reduce(ints, dm)
    del ints

    ints = _int3c(mol, coords, widths, 'int3c1e_ipvip1', 9, l, coeffs)
    for B, (p0, p1) in enumerate(aoslice[:, 2:4]):
        if p1 > p0:
            M[:, B] += reduce(ints[:, :, p0:p1], dm[p0:p1])
    del ints

    # ipip1 = ∂_i∂_i, ipvip1 = ∂_i∂_j (two sign flips); factor 2 for i <-> j
    M *= 2.0
    return _project2(M, O).reshape(natm, natm, 3, 3, ng)


def _project1(t, O):
    """Add the Gaussian-center term: d/dR_A → t_A - δ_{A,K(g)} Σ_B t_B."""
    return t - O[:, None, :] * t.sum(axis=0)


def _project2(M, O):
    """Second-derivative analogue of _project1 for M[A, B, comp, g]."""
    rows = M.sum(axis=1)
    cols = M.sum(axis=0)
    total = rows.sum(axis=0)
    return (M
            - O[None, :, None, :] * rows[:, None]
            - O[:, None, None, :] * cols[None, :]
            + (O[:, None, None, :] * O[None, :, None, :]) * total)


def _outer(a, b):
    """a[A,x,g] · b[B,y,g] → (A, B, x, y, g)."""
    return a[:, None, :, None, :] * b[None, :, None, :, :]


def _outer_sum(a, b):
    """Σ_g a[A,x,g] · b[B,y,g] → (A, B, x, y)."""
    return np.einsum('Axg,Byg->ABxy', a, b)


# ---------------------------------------------------------------------------
# First-order quantities shared by traces, d²e and d²F
# ---------------------------------------------------------------------------

def _first_order_terms(gost, dm):
    """Per-grid traces and their position derivatives (at fixed width).

    Keys:
      e_pos, F_pos   : ∂e/∂R, ∂F/∂R                          (natm, 3, ng)
      r2_pos, fr2_pos: ∂/∂R of Tr[D r²G̃] and f_contracted     (natm, 3, ng)
      r2, r4         : Tr[D r² G̃], Tr[D r⁴ G̃]                 (ng,)
      fr2, fr4       : f_contracted, h5 (p-type r², r⁴ traces) (ng,)
    """
    mol = gost.mol
    dm = _symmetrize(dm)
    natm = mol.natm
    ngrids = gost.n_gaussian
    U = _ao_to_atom(mol)
    O = _grid_to_atom(gost)

    out = {k: np.zeros((natm, 3, ngrids))
           for k in ('e_pos', 'F_pos', 'r2_pos', 'fr2_pos')}
    out.update({k: np.zeros(ngrids) for k in ('r2', 'r4', 'fr2', 'fr4')})

    # Largest live buffer: p-type ip1, 3 comps x 3 aux comps per grid point
    for g0, g1 in _grid_chunks(gost, 9):
        X = gost.grid_coords[g0:g1]
        w = gost.widths[g0:g1]
        n = gost.surface_normals[g0:g1]
        Oc = O[:, g0:g1]
        sl = slice(g0, g1)

        # s-type: Gtilde and its r², r⁴ moments
        out['e_pos'][..., sl] = _deriv1(
            _int3c(mol, X, w, 'int3c1e_ip1', 3), dm, U, Oc)
        out['r2_pos'][..., sl] = _deriv1(
            _int3c(mol, X, w, 'int3c1e_ip1_r2_origk', 3), dm, U, Oc)
        out['r2'][sl] = _trace(_int3c(mol, X, w, 'int3c1e_r2_origk', 1), dm)
        out['r4'][sl] = _trace(_int3c(mol, X, w, 'int3c1e_r4_origk', 1), dm)

        # p-type: Fhat (coeff 2ω) and its width moments
        out['F_pos'][..., sl] = _deriv1(
            _int3c(mol, X, w, 'int3c1e_ip1', 3, l=1, coeffs=2.0 * w),
            dm, U, Oc, normals=n)
        # ∂Fhat/∂ω = Fhat/ω + f_contracted, with f = -2ω · r² p-type
        out['fr2_pos'][..., sl] = _deriv1(
            _int3c(mol, X, w, 'int3c1e_ip1_r2_origk', 3, l=1, coeffs=-2.0 * w),
            dm, U, Oc, normals=n)
        out['fr2'][sl] = _trace(
            _int3c(mol, X, w, 'int3c1e_r2_origk', 1, l=1, coeffs=-2.0 * w),
            dm, normals=n)
        # ∂²Fhat/∂ω² = 2 f_contracted/ω + h5, with h5 = 2ω · r⁴ p-type
        out['fr4'][sl] = _trace(
            _int3c(mol, X, w, 'int3c1e_r4_origk', 1, l=1, coeffs=2.0 * w),
            dm, normals=n)
    return out


# ---------------------------------------------------------------------------
# First derivatives of scalar traces
# ---------------------------------------------------------------------------

def _compute_scalar_traces(gost, dm, first=None, dA=None):
    """Compute de_g/dR_Ax and dF_g/dR_Ax for all atoms.

    Returns
    -------
    dg_trace : ndarray (natm, 3, ngrids)
    dF_trace : ndarray (natm, 3, ngrids)
    """
    if first is None:
        first = _first_order_terms(gost, dm)
    if dA is None:
        dA = _area_derivatives(gost)[0]
    wgrad_prefs = -np.pi * np.log(2) / (gost.areas ** 2)  # dω/dA

    # Width contribution: dω/dR · ∂e/∂ω with ∂e/∂ω = -Tr[D r² G̃]
    dg_trace = first['e_pos'] - dA * (wgrad_prefs * first['r2'])
    # Width contribution: dω/dR · ∂F/∂ω with ∂F/∂ω = F/ω + f_contracted
    dFhat_domega = gost.forces / gost.widths + first['fr2']
    dF_trace = first['F_pos'] + dA * (wgrad_prefs * dFhat_domega)
    return dg_trace, dF_trace


# ---------------------------------------------------------------------------
# d²e_g/(dR_Ax dR_By) — second derivative of Gtilde trace
# ---------------------------------------------------------------------------

def _compute_d2e(gost, dm, first=None, dA=None, d2A=None):
    """Compute d²e_g/(dR_Ax dR_By) analytically.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    mol = gost.mol
    dm = _symmetrize(dm)
    if first is None:
        first = _first_order_terms(gost, dm)
    if dA is None or d2A is None:
        dA, d2A = _area_derivatives(gost)
    areas = gost.areas
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)
    U = _ao_to_atom(mol)
    O = _grid_to_atom(gost)

    # === G1: Position × Position ===
    d2e = np.empty((mol.natm, mol.natm, 3, 3, gost.n_gaussian))
    for g0, g1 in _grid_chunks(gost, 9):
        d2e[..., g0:g1] = _deriv2(mol, gost.grid_coords[g0:g1],
                                  gost.widths[g0:g1], dm, U, O[:, g0:g1])

    # === G2+G3: Position × Width cross terms ===
    # ∂²e/(∂pos ∂ω) = -∂(Tr[D r² G̃])/∂pos
    wdA = -wgrad_prefs * dA
    d2e += _outer(first['r2_pos'], wdA) + _outer(wdA, first['r2_pos'])

    # === G4: d²ω · ∂e/∂ω, ∂e/∂ω = -r2 ===
    # d²ω = -2·wgp/A · dA dA + wgp · d²A
    d2e += _outer(dA, dA * (2.0 * wgrad_prefs / areas * first['r2']))
    d2e -= d2A * (wgrad_prefs * first['r2'])

    # === G5: dω_Ax · dω_By · ∂²e/∂ω², ∂²e/∂ω² = r4 ===
    d2e += _outer(wgrad_prefs * dA, wgrad_prefs * dA * first['r4'])
    return d2e


# ---------------------------------------------------------------------------
# d²F_g/(dR_Ax dR_By) — second derivative of Fhat (force) trace
# ---------------------------------------------------------------------------

def _compute_d2F(gost, dm, first=None, dA=None, d2A=None):
    """Compute d²F_g/(dR_Ax dR_By) analytically.

    Returns shape (natm, natm, 3, 3, ngrids).
    """
    mol = gost.mol
    dm = _symmetrize(dm)
    if first is None:
        first = _first_order_terms(gost, dm)
    if dA is None or d2A is None:
        dA, d2A = _area_derivatives(gost)
    areas = gost.areas
    widths = gost.widths
    forces = gost.forces
    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)
    U = _ao_to_atom(mol)
    O = _grid_to_atom(gost)

    # === G1: Position × Position (p-type + normals) ===
    d2F = np.empty((mol.natm, mol.natm, 3, 3, gost.n_gaussian))
    for g0, g1 in _grid_chunks(gost, 27):
        w = widths[g0:g1]
        d2F[..., g0:g1] = _deriv2(mol, gost.grid_coords[g0:g1], w, dm, U,
                                  O[:, g0:g1], l=1, coeffs=2.0 * w,
                                  normals=gost.surface_normals[g0:g1])

    # === G2+G3: Position × Width cross terms ===
    # ∂²F/(∂pos_Ax ∂ω) = (1/ω)·∂F/∂pos + ∂f_contracted/∂pos
    d_dFhat_domega_pos = first['F_pos'] / widths + first['fr2_pos']
    wdA = wgrad_prefs * dA
    d2F += _outer(d_dFhat_domega_pos, wdA) + _outer(wdA, d_dFhat_domega_pos)

    # === G4: d²ω · ∂F/∂ω ===
    dFhat_domega = forces / widths + first['fr2']
    d2F += _outer(dA, dA * (-2.0 * wgrad_prefs / areas * dFhat_domega))
    d2F += d2A * (wgrad_prefs * dFhat_domega)

    # === G5: dω_Ax · dω_By · ∂²F/∂ω² ===
    # ∂²F/∂ω² = (2/ω)·f_contracted + h5
    d2Fhat_domega2 = 2.0 * first['fr2'] / widths + first['fr4']
    d2F += _outer(wdA, wdA * d2Fhat_domega2)
    return d2F
