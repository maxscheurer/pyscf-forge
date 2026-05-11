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

import unittest
import numpy as np
from pyscf import gto, scf
from pyscf.solvent.gostshyp import GOSTSHYP, gostshyp_for_scf, compute_surface_normals


def make_hf_mol():
    return gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g', cart=True, verbose=0)


class TestGOSTSHYP_VDW(unittest.TestCase):
    """Tests with plain vdW cavity."""

    @classmethod
    def setUpClass(cls):
        cls.mol = make_hf_mol()
        cls.gost = GOSTSHYP(cls.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2})
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
            'npoints': 110, 'scaling_factor': 1.2, 'direct': True})
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
            'npoints': 110, 'scaling_factor': 1.2, 'direct': True})
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
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
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
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
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
        fd_grad = finite_diff.kernel(mf, displacement=1e-3)
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


if __name__ == '__main__':
    unittest.main()
