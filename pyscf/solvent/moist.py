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
MOIST cavity construction wrapper for PySCF.

Provides DROPSvdW (smooth, fully differentiable van der Waals) surface
construction via the MOIST library (Modular and Open-source Implicit
Solvation Toolkit).

References:
    L. Wittmann and A. Pausch, ChemRxiv, 2026.
    DOI: 10.26434/chemrxiv.15003893/v2

    https://github.com/lukaswittmann/moist
"""

import numpy as np

try:
    from moist import CavityDROPSvdW, Structure
    HAS_MOIST = True
except ImportError:
    HAS_MOIST = False


def _require_moist():
    """Raise ImportError if MOIST is not available."""
    if not HAS_MOIST:
        raise ImportError(
            "MOIST library is required for DROP cavity construction. "
            "See https://github.com/lukaswittmann/moist for installation.")


def build_drop_cavity(mol, nleb=110, **kwargs):
    """Build a DROPSvdW cavity from a PySCF Mol object.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object (coordinates must be in Bohr).
    nleb : int
        Number of Lebedev grid points per atomic sphere (default: 110).
    **kwargs
        Additional keyword arguments passed to CavityDROPSvdW constructor
        (e.g. blend_k, blend_1b, blend_2b, blend_3b, proj_level, etc.).

    Returns
    -------
    cavity : moist.CavityDROPSvdW
        The constructed cavity object with surface data.
    """
    _require_moist()

    numbers = mol.atom_charges()
    positions = mol.atom_coords()  # Bohr, same as MOIST
    structure = Structure(numbers, positions)

    cavity = CavityDROPSvdW(nleb=nleb, **kwargs)
    cavity.update(structure)

    return cavity


def get_surface_data(cavity):
    """Extract surface data from a MOIST cavity in PySCF-compatible format.

    Parameters
    ----------
    cavity : moist.CavityDROPSvdW
        A constructed MOIST cavity (after update()).

    Returns
    -------
    grid_coords : ndarray of shape (ngrid, 3)
        Surface grid point coordinates in Bohr.
    areas : ndarray of shape (ngrid,)
        Surface element areas.
    owner : ndarray of shape (ngrid,)
        Atom index for each grid point.
    """
    grid_coords = np.ascontiguousarray(cavity.xyz.T)   # (3, ngrid) -> (ngrid, 3)
    areas = np.ascontiguousarray(cavity.a)              # (ngrid,)
    owner = np.ascontiguousarray(cavity.owner.astype(np.int32))  # (ngrid,)

    return grid_coords, areas, owner


def get_anchor_gradient(cavity):
    """Compute and return the anchor gradient from a MOIST cavity.

    Triggers anchor gradient computation and returns the derivatives
    of grid-point areas and positions w.r.t. nuclear coordinates.

    Parameters
    ----------
    cavity : moist.CavityDROPSvdW
        A constructed MOIST cavity (after update()).

    Returns
    -------
    dareas : ndarray of shape (natm, 3, ngrid)
        d(a_i) / d(R_A)_alpha — per-point area derivatives.
    dcoords : ndarray of shape (3, 3, natm, ngrid)
        d(r_i)_j / d(R_A)_alpha — per-point position derivatives.
        Indices: (j=coord_of_grid, alpha=coord_of_atom, A=atom, i=grid).
    """
    cavity.compute_anchor_gradient()
    grad = cavity.get_anchor_gradient()

    # a_i1_rA: (alpha, atom, grid) -> (atom, alpha, grid)
    dareas = np.ascontiguousarray(grad.a_i1_rA.transpose(1, 0, 2))

    # xyz1_rA: (j, alpha, atom, grid) — keep as-is for chain rule
    dcoords = np.ascontiguousarray(grad.xyz1_rA)

    return dareas, dcoords
