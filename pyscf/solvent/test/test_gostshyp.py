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
from pyscf.solvent.gostshyp import (
    GOSTSHYP, gostshyp_for_scf, compute_surface_normals, analytical_grad_vmat
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


def _fd_grad_vmat(gost, dm, atmlst=None, eps=1e-5):
    """Finite-difference reference for dV/dR at fixed density.

    Parameters
    ----------
    gost : GOSTSHYP
        GOSTSHYP object (defines cavity parameters).
    dm : ndarray of shape (nao, nao)
        Density matrix held fixed.
    atmlst : list of int, optional
        Atoms for which to compute derivatives. Default: all.
    eps : float
        Finite-difference step size.

    Returns
    -------
    dV_fd : ndarray of shape (len(atmlst), 3, nao, nao)
    """
    mol = gost.mol
    if atmlst is None:
        atmlst = list(range(mol.natm))
    nao = mol.nao_nr()
    dV = np.zeros((len(atmlst), 3, nao, nao))

    options = {
        'cavity': gost.cavity,
        'pressure_mpa': gost.pressure_mpa,
        'npoints': gost.npoints,
        'scaling_factor': gost.scaling_factor,
    }
    if gost.cavity == 'vdw/occ':
        options['r_ext'] = gost.r_ext

    for ia, atm in enumerate(atmlst):
        for x in range(3):
            coords_p = mol.atom_coords().copy()
            coords_p[atm, x] += eps
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=options)
            _, v_p = gost_p.kernel(dm)

            coords_m = mol.atom_coords().copy()
            coords_m[atm, x] -= eps
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=options)
            _, v_m = gost_m.kernel(dm)

            dV[ia, x] = (v_p - v_m) / (2 * eps)
    return dV


class TestGradVmat(unittest.TestCase):
    """Tests for analytical_grad_vmat (dV/dR at fixed density)."""

    @classmethod
    def setUpClass(cls):
        cls.mol_hf = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g',
                            cart=True, verbose=0)
        cls.mol_h2o = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='6-31g', cart=True, verbose=0)

    def test_shape(self):
        """Output shape is (natm, 3, nao, nao)."""
        gost = GOSTSHYP(self.mol_hf, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(self.mol_hf)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        dV = analytical_grad_vmat(gost, dm)
        self.assertEqual(dV.shape, (2, 3, 11, 11))

    def test_symmetry(self):
        """Each dV[ia, x] slice must be symmetric."""
        gost = GOSTSHYP(self.mol_hf, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(self.mol_hf)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        dV = analytical_grad_vmat(gost, dm)
        for ia in range(2):
            for x in range(3):
                np.testing.assert_allclose(
                    dV[ia, x], dV[ia, x].T, atol=1e-14,
                    err_msg=f'dV[{ia},{x}] not symmetric')

    def test_fd_hf_molecule(self):
        """Analytic vs FD for HF molecule (cart=True, vdw cavity)."""
        gost = GOSTSHYP(self.mol_hf, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2})
        mf = scf.RHF(self.mol_hf)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        dV = analytical_grad_vmat(gost, dm)
        dV_fd = _fd_grad_vmat(gost, dm)
        np.testing.assert_allclose(dV, dV_fd, atol=1e-7)

    def test_fd_h2o(self):
        """Analytic vs FD for H2O (multi-atom, asymmetric)."""
        gost = GOSTSHYP(self.mol_h2o, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2})
        mf = scf.RHF(self.mol_h2o)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        dV = analytical_grad_vmat(gost, dm)
        dV_fd = _fd_grad_vmat(gost, dm)
        np.testing.assert_allclose(dV, dV_fd, atol=1e-7)

    def test_fd_spherical_basis(self):
        """Analytic vs FD with spherical harmonics (exercises c2s transform)."""
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='6-31g',
                    cart=False, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        dV = analytical_grad_vmat(gost, dm)
        dV_fd = _fd_grad_vmat(gost, dm)
        np.testing.assert_allclose(dV, dV_fd, atol=1e-7)

    def test_fd_vdw_occ(self):
        """Analytic vs FD with vdw/occ cavity."""
        gost = GOSTSHYP(self.mol_hf, options={
            'cavity': 'vdw/occ', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(self.mol_hf)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        dV = analytical_grad_vmat(gost, dm)
        dV_fd = _fd_grad_vmat(gost, dm)
        np.testing.assert_allclose(dV, dV_fd, atol=1e-7)

    def test_atmlst_subset(self):
        """Test with atmlst selecting a subset of atoms."""
        gost = GOSTSHYP(self.mol_h2o, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 110})
        mf = scf.RHF(self.mol_h2o)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        atmlst = [0, 2]
        dV_sub = analytical_grad_vmat(gost, dm, atmlst=atmlst)
        dV_all = analytical_grad_vmat(gost, dm)
        self.assertEqual(dV_sub.shape, (2, 3, 13, 13))
        np.testing.assert_allclose(dV_sub[0], dV_all[0], atol=1e-14)
        np.testing.assert_allclose(dV_sub[1], dV_all[2], atol=1e-14)

    def test_fd_masked_amplitudes(self):
        """Analytic vs FD with tight cavity that triggers amplitude masking."""
        gost = GOSTSHYP(self.mol_hf, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 0.5})
        mf = scf.RHF(self.mol_hf)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        # Verify masking is actually triggered
        self.assertTrue(np.any(gost.amplitudes == 0.0),
                        'Test requires masked amplitudes but none were masked')

        dV = analytical_grad_vmat(gost, dm)
        dV_fd = _fd_grad_vmat(gost, dm)
        np.testing.assert_allclose(dV, dV_fd, atol=1e-7)

    def test_kernel_not_called_raises(self):
        """Must call kernel() before analytical_grad_vmat."""
        gost = GOSTSHYP(self.mol_hf, options={'cavity': 'vdw'})
        dm = np.eye(self.mol_hf.nao_nr())
        with self.assertRaises(RuntimeError):
            analytical_grad_vmat(gost, dm)


if __name__ == '__main__':
    unittest.main()
