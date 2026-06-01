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

import os
import unittest
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.solvent.gostshyp import (
    GOSTSHYP, gostshyp_for_scf, compute_surface_normals, analytical_grad_vmat,
    WithGOSTSHYPHess, make_hess_object
)


def make_hf_mol():
    return gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True, verbose=0)


class TestGOSTSHYP_VDW(unittest.TestCase):
    """Tests with plain vdW cavity."""

    @classmethod
    def setUpClass(cls):
        cls.mol = make_hf_mol()
        cls.gost = GOSTSHYP(cls.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2, 'area_thresh': None})
        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.conv_tol_grad = 1e-8
        cls.mf = gostshyp_for_scf(cls.mf, cls.gost)
        cls.mf.kernel()

    def test_rhf_energy(self):
        self.assertAlmostEqual(self.mf.e_tot, -99.8941733641653, places=7)

    def test_converged(self):
        self.assertTrue(self.mf.converged)

    def test_fock_symmetry(self):
        dm = self.mf.make_rdm1()
        _, fock = self.gost.kernel(dm)
        np.testing.assert_allclose(fock, fock.T, atol=1e-12)

    def test_gradient(self):
        dm = self.mf.make_rdm1()
        grad = self.gost.grad(dm)
        ref = np.array([[-0.00951946951, 0.0, 0.0],
                        [ 0.00951946951, 0.0, 0.0]])
        np.testing.assert_allclose(grad, ref, atol=1e-7)

    def test_translational_invariance(self):
        dm = self.mf.make_rdm1()
        grad = self.gost.grad(dm)
        np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-8)

    def test_direct_vs_cached(self):
        dm = self.mf.make_rdm1()
        e_cached, f_cached = self.gost.kernel(dm)

        gost_direct = GOSTSHYP(self.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2, 'direct': True, 'area_thresh': None})
        e_direct, f_direct = gost_direct.kernel(dm)

        np.testing.assert_allclose(e_direct, e_cached, atol=1e-12)
        np.testing.assert_allclose(f_direct, f_cached, atol=1e-12)


class TestSurfaceNormals(unittest.TestCase):
    def test_unit_normals(self):
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={'cavity': 'vdw'})
        norms = np.linalg.norm(gost.surface_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)


class TestGOSTSHYP_UHF(unittest.TestCase):
    def test_uhf_converges(self):
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={'cavity': 'vdw', 'pressure_mpa': 50_000})
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-10
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)


class TestGOSTSHYP_VDW_OCC(unittest.TestCase):
    def test_vdw_occ_converges(self):
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw/occ', 'pressure_mpa': 50_000})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)

    def test_vdw_occ_smaller_area(self):
        """vdW/OCC should have smaller effective areas than plain vdW."""
        mol = make_hf_mol()
        gost_vdw = GOSTSHYP(mol, options={'cavity': 'vdw'})
        gost_occ = GOSTSHYP(mol, options={'cavity': 'vdw/occ'})
        self.assertLess(gost_occ.areas.sum(), gost_vdw.areas.sum())


class TestFiniteDifference(unittest.TestCase):
    def test_gradient_finite_difference(self):
        """Full SCF+GOSTSHYP analytic gradient vs finite_diff."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()

        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestDirectGradient(unittest.TestCase):
    def test_direct_gradient_vs_cached(self):
        """Direct gradient must match cached gradient."""
        mol = make_hf_mol()
        opts = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}

        gost_c = GOSTSHYP(mol, options={**opts, 'direct': False})
        gost_d = GOSTSHYP(mol, options={**opts, 'direct': True})

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost_c)
        mf.kernel()
        dm = mf.make_rdm1()

        gost_c.kernel(dm)
        gost_d.kernel(dm)

        grad_cached = gost_c.grad(dm)
        grad_direct = gost_d.grad(dm)
        np.testing.assert_allclose(grad_direct, grad_cached, atol=1e-12)

    def test_direct_gradient_reference(self):
        """Direct gradient must match known reference values."""
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2, 'direct': True, 'area_thresh': None})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        grad = gost.grad(dm)
        ref = np.array([[-0.00951946951, 0.0, 0.0],
                        [ 0.00951946951, 0.0, 0.0]])
        np.testing.assert_allclose(grad, ref, atol=1e-7)

    def test_direct_gradient_translational_invariance(self):
        """Direct gradient must sum to zero over atoms."""
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'direct': True})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        grad = gost.grad(dm)
        np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-8)


class TestSCFAttachment(unittest.TestCase):
    def test_mf_gostshyp_method(self):
        """Test that mf.GOSTSHYP() works after import."""
        mol = make_hf_mol()
        mf = scf.RHF(mol)
        mf_sol = mf.GOSTSHYP()
        self.assertTrue(hasattr(mf_sol, 'with_solvent'))

    def test_gostshyp_rejects_post_scf(self):
        from pyscf import mp
        mol = make_hf_mol()
        mf = scf.RHF(mol).run()
        mp2 = mp.MP2(mf)
        with self.assertRaises(TypeError):
            gostshyp_for_scf(mp2)


class TestFiniteDifferenceOCC(unittest.TestCase):
    def test_gradient_finite_difference_occ(self):
        """Full SCF+GOSTSHYP analytic gradient vs finite_diff for vdw/occ cavity."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw/occ', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-4)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestSphericalHarmonics(unittest.TestCase):
    def test_gradient_spherical_basis(self):
        """Gradient with spherical harmonic basis (exercises c2s transform)."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=False, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-4)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestMultiAtom(unittest.TestCase):
    def test_water_gradient(self):
        """Gradient for a non-linear molecule (exercises np.add.at accumulation)."""
        from pyscf.tools import finite_diff
        mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                    basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-4)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestNegativeAmplitudeMasking(unittest.TestCase):
    """Tight cavity triggers negative amplitudes; verify they are masked."""

    def test_no_negative_amplitudes(self):
        mol = make_hf_mol()
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 0.5})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertTrue(np.all(mf.with_solvent.amplitudes >= 0))

    def test_gradient_finite_difference_tight_cavity(self):
        from pyscf.tools import finite_diff
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 26, 'scaling_factor': 0.5})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        analytic = mf.Gradients().kernel()
        fd_grad = finite_diff.kernel(mf, displacement=1e-4)
        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestReset(unittest.TestCase):
    def test_reset_rebuilds_surface(self):
        """reset() clears cached properties and rebuilds for new geometry."""
        mol1 = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(mol1, options={'cavity': 'vdw', 'pressure_mpa': 50_000})
        mf = scf.RHF(mol1)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        e1 = mf.e_tot

        mol2 = gto.M(atom='H 1 0 0; F 2.2 0 0', basis='sto-3g', cart=True, verbose=0)
        mf.reset(mol2)
        mf.kernel()
        e2 = mf.e_tot
        self.assertNotAlmostEqual(e1, e2, places=5)
        self.assertTrue(mf.converged)


# ============================================================
# Hessian test utilities
# ============================================================

from pyscf.solvent.test.reference_systems import SYSTEMS

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference_data')
SYSTEM_NAMES = list(SYSTEMS.keys())


def _gost_options(gost):
    """Extract GOSTSHYP options dict for reconstruction at displaced geometry."""
    opts = {
        'cavity': gost.cavity, 'pressure_mpa': gost.pressure_mpa,
        'npoints': gost.npoints, 'scaling_factor': gost.scaling_factor,
    }
    if gost.cavity == 'vdw/occ':
        opts['r_ext'] = gost.r_ext
    return opts


def _make_gost(system_name):
    """Build GOSTSHYP with converged density for a reference system.

    Returns (gost, dm, mol).
    """
    system = SYSTEMS[system_name]
    mol = gto.M(atom=system.atom, basis=system.basis,
                unit=system.unit, verbose=0)
    gost = GOSTSHYP(mol, options=system.gostshyp.to_dict())
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()
    gost.kernel(dm)
    return gost, dm, mol


def _fd_over_geometry(gost, dm, eval_fn, step=1e-5):
    """Central finite difference of eval_fn(gost, dm) w.r.t. nuclear coords.

    Parameters
    ----------
    gost : GOSTSHYP with kernel() already called
    dm : density matrix (held fixed)
    eval_fn : callable(gost, dm) -> ndarray of any shape
    step : displacement in Bohr

    Returns
    -------
    deriv : ndarray of shape (natm, 3, *eval_shape)
    """
    mol = gost.mol
    natm = mol.natm
    coords0 = mol.atom_coords().copy()
    opts = _gost_options(gost)

    # First displacement to infer output shape
    coords_p = coords0.copy()
    coords_p[0, 0] += step
    mol_p = mol.copy()
    mol_p.set_geom_(coords_p, unit='Bohr')
    gost_p = GOSTSHYP(mol_p, options=opts)
    gost_p.kernel(dm)
    ref = eval_fn(gost_p, dm)
    result = np.zeros((natm, 3, *ref.shape))

    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy()
            coords_p[B, y] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=opts)
            gost_p.kernel(dm)
            val_p = eval_fn(gost_p, dm)

            coords_m = coords0.copy()
            coords_m[B, y] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=opts)
            gost_m.kernel(dm)
            val_m = eval_fn(gost_m, dm)

            result[B, y] = (val_p - val_m) / (2.0 * step)

    return result


# ============================================================
# analytical_grad_vmat tests
# ============================================================

@pytest.mark.parametrize('system_name', SYSTEM_NAMES)
def test_grad_vmat_vs_fd(system_name):
    """analytical_grad_vmat matches fdiff of kernel().v."""
    gost, dm, mol = _make_gost(system_name)
    dV_ana = analytical_grad_vmat(gost, dm)
    dV_fd = _fd_over_geometry(gost, dm, lambda g, d: g.v)
    np.testing.assert_allclose(
        dV_ana, dV_ana.transpose(0, 1, 3, 2), atol=1e-14,
        err_msg=f'dV not symmetric'
    )
    np.testing.assert_allclose(dV_ana, dV_fd, atol=1e-7)


def test_grad_vmat_atmlst_subset():
    """atmlst selects a subset of atoms."""
    gost, dm, mol = _make_gost('h2o_cc-pvdz')
    atmlst = [0, 2]
    dV_sub = analytical_grad_vmat(gost, dm, atmlst=atmlst)
    dV_all = analytical_grad_vmat(gost, dm)
    assert dV_sub.shape[0] == 2
    np.testing.assert_allclose(dV_sub[0], dV_all[0], atol=1e-14)
    np.testing.assert_allclose(dV_sub[1], dV_all[2], atol=1e-14)


def test_grad_vmat_vdw_occ():
    """Analytic vs FD with vdw/occ cavity."""
    mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True, verbose=0)
    gost = GOSTSHYP(mol, options={
        'cavity': 'vdw/occ', 'pressure_mpa': 50_000, 'npoints': 110})
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf = gostshyp_for_scf(mf, gost)
    mf.kernel()
    dm = mf.make_rdm1()
    gost.kernel(dm)
    dV_ana = analytical_grad_vmat(gost, dm)
    dV_fd = _fd_over_geometry(gost, dm, lambda g, d: g.v)
    np.testing.assert_allclose(dV_ana, dV_fd, atol=1e-7)


def test_grad_vmat_masked_amplitudes():
    """Analytic vs FD with tight cavity that triggers amplitude masking."""
    mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True, verbose=0)
    gost = GOSTSHYP(mol, options={
        'cavity': 'vdw', 'pressure_mpa': 50_000,
        'npoints': 110, 'scaling_factor': 0.5, 'area_thresh': None})
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf = gostshyp_for_scf(mf, gost)
    mf.kernel()
    dm = mf.make_rdm1()
    gost.kernel(dm)
    assert np.any(gost.amplitudes == 0.0), 'No masked amplitudes'
    dV_ana = analytical_grad_vmat(gost, dm)
    dV_fd = _fd_over_geometry(gost, dm, lambda g, d: g.v)
    np.testing.assert_allclose(dV_ana, dV_fd, atol=1e-7)


# ============================================================
# d2e / d2F tests (unified)
# ============================================================

@pytest.mark.parametrize('system_name', SYSTEM_NAMES)
@pytest.mark.parametrize('which', ['d2e', 'd2F'])
def test_d2_scalar_vs_fd(system_name, which):
    """d2e/d2F analytical matches fdiff of first-order traces."""
    from pyscf.solvent._gostshyp_hess import (
        _compute_d2e, _compute_d2F, _compute_scalar_traces)

    gost, dm, _ = _make_gost(system_name)

    if which == 'd2e':
        d2_ana = _compute_d2e(gost, dm)
        eval_fn = lambda g, d: _compute_scalar_traces(g, d)[0]
    else:
        d2_ana = _compute_d2F(gost, dm)
        eval_fn = lambda g, d: _compute_scalar_traces(g, d)[1]
    
    np.testing.assert_allclose(d2_ana, d2_ana.transpose(1, 0, 3, 2, 4), atol=1e-10,
                               err_msg=f'{which} not symmetric in (A,x) <-> (B,y)')

    # fdiff returns (B, y, A, x, g); analytical is (A, B, x, y, g)
    d2_fd_raw = _fd_over_geometry(gost, dm, eval_fn, step=1e-4)
    d2_fd = d2_fd_raw.transpose(2, 0, 3, 1, 4)
    np.testing.assert_allclose(d2_ana, d2_fd, atol=1e-6,
                               err_msg=f'{which} analytical vs fd mismatch')


# ============================================================
# Hessian infrastructure (keep as unittest — tests specific wiring)
# ============================================================

class TestHessianInfrastructure(unittest.TestCase):
    """Tests for the GOSTSHYP Hessian hook and infrastructure."""

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g',
                        cart=True, verbose=0)
        cls.gost = GOSTSHYP(cls.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf = gostshyp_for_scf(cls.mf, cls.gost)
        cls.mf.kernel()

    def test_hessian_runs_and_shape(self):
        hess = self.mf.Hessian().kernel()
        self.assertEqual(hess.shape, (2, 2, 3, 3))

    def test_hess_returns_nonzero(self):
        dm = self.mf.make_rdm1()
        de_solvent = self.gost.hess(dm)
        self.assertEqual(de_solvent.shape, (2, 2, 3, 3))
        self.assertGreater(np.max(np.abs(de_solvent)), 1e-6)

    def test_make_h1_includes_grad_vmat(self):
        hess_obj = self.mf.Hessian()
        h1_sol = hess_obj.make_h1(self.mf.mo_coeff, self.mf.mo_occ)
        vac_hess = hess_obj.undo_solvent()
        h1_vac = vac_hess.make_h1(self.mf.mo_coeff, self.mf.mo_occ)
        dm = self.mf.make_rdm1()
        self.gost.kernel(dm)
        dv = analytical_grad_vmat(self.gost, dm)
        for ia in range(self.mol.natm):
            np.testing.assert_allclose(h1_sol[ia] - h1_vac[ia], dv[ia], atol=1e-12)

    def test_hessian_hook_type(self):
        self.assertIsInstance(self.mf.Hessian(), WithGOSTSHYPHess)

    def test_uhf_hessian_runs(self):
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g',
                    cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertEqual(hess.shape, (2, 2, 3, 3))


# ============================================================
# gost.hess(dm) vs gost.hess_fd(dm) — direct term at fixed density
# ============================================================

@pytest.mark.parametrize('system_name', SYSTEM_NAMES)
def test_hess_vs_fd(system_name):
    """gost.hess(dm) matches gost.hess_fd(dm) at fixed density."""
    gost, dm, mol = _make_gost(system_name)
    H_ana = gost.hess(dm)
    H_fd = gost.hess_fd(dm, step=1e-4)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-6)


def test_hess_vs_fd_masked():
    """hess(dm) matches hess_fd(dm) with tight cavity (amplitude masking)."""
    mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='cc-pVDZ',
                cart=True, verbose=0)
    gost = GOSTSHYP(mol, options={
        'cavity': 'vdw', 'pressure_mpa': 50_000,
        'npoints': 110, 'scaling_factor': 0.5, 'area_thresh': None})
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf = gostshyp_for_scf(mf, gost)
    mf.kernel()
    dm = mf.make_rdm1()
    gost.kernel(dm)
    assert np.any(gost.amplitudes == 0.0), 'No masked amplitudes'
    H_ana = gost.hess(dm)
    H_fd = gost.hess_fd(dm, step=1e-4)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-6)


# ============================================================
# End-to-end Hessian vs stored numerical references
# ============================================================

@pytest.mark.parametrize('system_name', SYSTEM_NAMES)
def test_e2e_hessian(system_name):
    """Full SCF+GOSTSHYP Hessian vs stored numerical reference."""
    from pyscf.solvent.test.reference_systems import make_mf

    ref_path = os.path.join(REFERENCE_DIR, f'hess_{system_name}.npy')
    if not os.path.exists(ref_path):
        pytest.skip(f'Reference not found. Run: python -m pyscf.solvent.test.'
                    f'generate_gostshyp_hessian_references --systems {system_name}')

    hess_ref = np.load(ref_path)
    mf = make_mf(system_name)
    hess_ana = mf.Hessian().kernel()
    np.testing.assert_allclose(hess_ana, hess_ref, atol=1e-6)


# ============================================================
# _B_dot_x — CPHF linear response kernel
# ============================================================

@pytest.mark.parametrize('system_name', SYSTEM_NAMES)
def test_B_dot_x_vs_fd(system_name):
    """_B_dot_x matches finite differences of kernel()."""
    gost, dm, mol = _make_gost(system_name)
    nao = mol.nao_nr()
    rng = np.random.default_rng(42)
    dm1 = rng.standard_normal((nao, nao))
    dm1 = 0.5 * (dm1 + dm1.T)

    v_ana = gost._B_dot_x(dm1)

    eps = 1e-5
    opts = _gost_options(gost)
    gost_p = GOSTSHYP(mol, options=opts)
    gost_p.kernel(dm + eps * dm1)
    gost_m = GOSTSHYP(mol, options=opts)
    gost_m.kernel(dm - eps * dm1)
    v_fd = (gost_p.v - gost_m.v) / (2 * eps)

    np.testing.assert_allclose(v_ana, v_fd, atol=1e-7)


@pytest.fixture
def b_dot_x_system():
    """H2O system for _B_dot_x property tests."""
    gost, dm, mol = _make_gost('h2o_cc-pvdz')
    rng = np.random.default_rng(42)
    nao = mol.nao_nr()
    dm1 = rng.standard_normal((nao, nao))
    dm1 = 0.5 * (dm1 + dm1.T)
    return gost, dm, mol, dm1


def test_B_dot_x_symmetry(b_dot_x_system):
    """_B_dot_x output is symmetric."""
    gost, dm, mol, dm1 = b_dot_x_system
    v = gost._B_dot_x(dm1)
    np.testing.assert_allclose(v, v.T, atol=1e-14)


def test_B_dot_x_direct_vs_cached(b_dot_x_system):
    """Direct and cached modes give same result."""
    gost, dm, mol, dm1 = b_dot_x_system
    opts = _gost_options(gost)
    opts['direct'] = False
    gost_cached = GOSTSHYP(mol, options=opts)
    gost_cached.kernel(dm)
    np.testing.assert_allclose(gost._B_dot_x(dm1), gost_cached._B_dot_x(dm1), atol=1e-12)


def test_B_dot_x_batched(b_dot_x_system):
    """Batched input gives same result as individual calls."""
    gost, dm, mol, dm1 = b_dot_x_system
    nao = mol.nao_nr()
    rng = np.random.default_rng(123)
    dm1_batch = rng.standard_normal((3, nao, nao))
    dm1_batch = 0.5 * (dm1_batch + dm1_batch.transpose(0, 2, 1))
    v_batch = gost._B_dot_x(dm1_batch)
    v_singles = np.array([gost._B_dot_x(dm1_batch[i]) for i in range(3)])
    np.testing.assert_allclose(v_batch, v_singles, atol=1e-14)


def test_B_dot_x_shape(b_dot_x_system):
    """Shape matches input."""
    gost, dm, mol, dm1 = b_dot_x_system
    nao = mol.nao_nr()
    assert gost._B_dot_x(dm1).shape == (nao, nao)
    assert gost._B_dot_x(np.zeros((4, nao, nao))).shape == (4, nao, nao)


if __name__ == '__main__':
    unittest.main()
