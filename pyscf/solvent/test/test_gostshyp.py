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
        ref_x = np.array([-0.00951946951, 0.00951946951])
        np.testing.assert_allclose(grad[:, 0], ref_x, atol=1e-7)
        # y and z should be ~0
        np.testing.assert_allclose(grad[:, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(grad[:, 2], 0.0, atol=1e-10)

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
        """Check gradient against central finite differences."""
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()

        analytic = gost.grad(dm)

        h = 1e-4
        natm = mol.natm
        fd_grad = np.zeros((natm, 3))
        for iatm in range(natm):
            for ix in range(3):
                coords_p = mol.atom_coords().copy()
                coords_m = mol.atom_coords().copy()
                coords_p[iatm, ix] += h
                coords_m[iatm, ix] -= h

                mol_p = mol.copy()
                mol_p.set_geom_(coords_p, unit='Bohr')
                gost_p = GOSTSHYP(mol_p, options={
                    'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
                ep, _ = gost_p.kernel(dm)

                mol_m = mol.copy()
                mol_m.set_geom_(coords_m, unit='Bohr')
                gost_m = GOSTSHYP(mol_m, options={
                    'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
                em, _ = gost_m.kernel(dm)

                fd_grad[iatm, ix] = (ep - em) / (2 * h)

        np.testing.assert_allclose(analytic, fd_grad, atol=1e-5)


class TestSCFAttachment(unittest.TestCase):
    def test_mf_gostshyp_method(self):
        """Test that mf.GOSTSHYP() works after import."""
        mol = make_hf_mol()
        mf = scf.RHF(mol)
        mf_sol = mf.GOSTSHYP()
        self.assertTrue(hasattr(mf_sol, 'with_solvent'))


if __name__ == '__main__':
    unittest.main()
