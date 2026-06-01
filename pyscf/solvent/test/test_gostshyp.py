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
        """GOSTSHYP Hessian hook runs end-to-end, returns correct shape."""
        hess = self.mf.Hessian().kernel()
        natm = self.mol.natm
        self.assertEqual(hess.shape, (natm, natm, 3, 3))

    def test_hess_returns_nonzero(self):
        """GOSTSHYP.hess(dm) returns non-zero values with correct shape."""
        dm = self.mf.make_rdm1()
        de_solvent = self.gost.hess(dm)
        natm = self.mol.natm
        self.assertEqual(de_solvent.shape, (natm, natm, 3, 3))
        self.assertGreater(np.max(np.abs(de_solvent)), 1e-6)

    def test_make_h1_includes_grad_vmat(self):
        """make_h1 augments vacuum h1 with analytical_grad_vmat."""
        hess_obj = self.mf.Hessian()
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        h1_sol = hess_obj.make_h1(mo_coeff, mo_occ)

        # Compare to vacuum make_h1
        vac_hess = hess_obj.undo_solvent()
        h1_vac = vac_hess.make_h1(mo_coeff, mo_occ)

        # Difference should be analytical_grad_vmat
        dm = self.mf.make_rdm1()
        self.gost.kernel(dm)
        dv = analytical_grad_vmat(self.gost, dm)

        for ia in range(self.mol.natm):
            diff = h1_sol[ia] - h1_vac[ia]
            np.testing.assert_allclose(diff, dv[ia], atol=1e-12,
                                       err_msg=f'make_h1 difference != grad_vmat for atom {ia}')

    def test_hessian_hook_type(self):
        """Hessian() returns WithGOSTSHYPHess instance."""
        hess_obj = self.mf.Hessian()
        self.assertIsInstance(hess_obj, WithGOSTSHYPHess)


class TestHessianUHF(unittest.TestCase):
    """Test GOSTSHYP Hessian with UHF."""

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


class TestExplicitHessian(unittest.TestCase):
    """Tests for GOSTSHYP.hess(dm) — explicit d²E/dR² at fixed density."""

    def _run_hess(self, mol, cavity='vdw', npoints=110, scaling_factor=1.2):
        """Helper: SCF → kernel → hess + hess_fd."""
        opts = {'cavity': cavity, 'pressure_mpa': 50_000,
                'npoints': npoints, 'scaling_factor': scaling_factor}
        if cavity == 'vdw/occ':
            opts['r_ext'] = 0.4724
        gost = GOSTSHYP(mol, options=opts)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        return gost, dm

    def test_hess_fd_shape(self):
        """hess_fd returns (natm, natm, 3, 3)."""
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g',
                    cart=True, verbose=0)
        gost, dm = self._run_hess(mol, npoints=26)
        H = gost.hess_fd(dm)
        self.assertEqual(H.shape, (2, 2, 3, 3))

    def test_hess_fd_symmetry(self):
        """hess_fd satisfies H[A,B,x,y] ~ H[B,A,y,x]."""
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g',
                    cart=True, verbose=0)
        gost, dm = self._run_hess(mol, npoints=26)
        H = gost.hess_fd(dm)
        np.testing.assert_allclose(H, H.transpose(1, 0, 3, 2), atol=1e-7)

    def test_hess_shape_h2(self):
        """hess(dm) returns correct shape for H2."""
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pVDZ',
                    cart=True, verbose=0)
        gost, dm = self._run_hess(mol, npoints=110)
        H = gost.hess(dm)
        self.assertEqual(H.shape, (2, 2, 3, 3))

    def test_hess_symmetry_h2(self):
        """Analytical Hessian is symmetric for H2."""
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pVDZ',
                    cart=True, verbose=0)
        gost, dm = self._run_hess(mol, npoints=110)
        H = gost.hess(dm)
        np.testing.assert_allclose(H, H.transpose(1, 0, 3, 2), atol=1e-7)

    def test_hess_vs_fd_masked(self):
        """hess(dm) matches hess_fd(dm) with tight cavity (amplitude masking)."""
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='cc-pVDZ',
                    cart=True, verbose=0)
        gost, dm = self._run_hess(mol, npoints=110, scaling_factor=0.5)
        # Verify masking is triggered
        self.assertTrue(np.any(gost.amplitudes == 0.0),
                        'Test requires masked amplitudes but none were masked')
        H_ana = gost.hess(dm)
        H_fd = gost.hess_fd(dm, step=1e-4)
        np.testing.assert_allclose(H_ana, H_fd, atol=1e-5)

    def test_hess_pipeline_end_to_end(self):
        """Full Hessian pipeline (mf.Hessian().kernel()) runs with hess(dm)."""
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g',
                    cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000, 'npoints': 26})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertEqual(hess.shape, (2, 2, 3, 3))
        # Should not be all zeros anymore (unlike the old stub)
        self.assertGreater(np.max(np.abs(gost.hess(mf.make_rdm1()))), 1e-6)

    def test_hess_kernel_not_called_raises(self):
        """Must call kernel() before hess()."""
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g',
                    cart=True, verbose=0)
        gost = GOSTSHYP(mol, options={'cavity': 'vdw'})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        # Reset forces to None to simulate kernel() not being called
        gost.forces = None
        with self.assertRaises(RuntimeError):
            gost.hess(dm)


def _fd_d2_scalar(gost, dm, compute_trace_fn, step=1e-4):
    """Finite-difference d²(scalar)/dR² by central diff of first derivatives.

    Parameters
    ----------
    gost : GOSTSHYP with kernel() called
    dm : density matrix (fixed)
    compute_trace_fn : callable(gost, dm) -> (natm, 3, ngrids)
        Returns the first-order trace (dg_trace or dF_trace).
    step : float

    Returns
    -------
    d2_fd : ndarray (natm, natm, 3, 3, ngrids)
    """
    from pyscf.solvent.gostshyp import GOSTSHYP
    from pyscf.solvent._gostshyp_hess import _compute_scalar_traces

    mol = gost.mol
    natm = mol.natm
    ngrids = gost.n_gaussian
    coords0 = mol.atom_coords().copy()
    opts = {
        'cavity': gost.cavity, 'pressure_mpa': gost.pressure_mpa,
        'npoints': gost.npoints, 'scaling_factor': gost.scaling_factor,
    }
    if gost.cavity == 'vdw/occ':
        opts['r_ext'] = gost.r_ext

    d2_fd = np.zeros((natm, natm, 3, 3, ngrids))
    for B in range(natm):
        for y in range(3):
            coords_p = coords0.copy()
            coords_p[B, y] += step
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            gost_p = GOSTSHYP(mol_p, options=opts)
            gost_p.kernel(dm)
            trace_p = compute_trace_fn(gost_p, dm)

            coords_m = coords0.copy()
            coords_m[B, y] -= step
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            gost_m = GOSTSHYP(mol_m, options=opts)
            gost_m.kernel(dm)
            trace_m = compute_trace_fn(gost_m, dm)

            d2_fd[:, B, :, y, :] = (trace_p - trace_m) / (2.0 * step)

    return d2_fd


class TestD2e(unittest.TestCase):
    """Tests for _compute_d2e — second derivative of Gtilde trace."""

    @classmethod
    def setUpClass(cls):
        from pyscf.solvent._gostshyp_hess import _compute_d2e, _compute_scalar_traces
        # N2/cc-pVDZ — clean system with large areas
        mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ',
                    unit='Angstrom', verbose=0)
        opts = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}
        gost = GOSTSHYP(mol, options=opts)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        cls.mol = mol
        cls.gost = gost
        cls.dm = dm

    def test_d2e_vs_fd_n2(self):
        """d2e analytical matches fdiff on N2/cc-pVDZ."""
        from pyscf.solvent._gostshyp_hess import _compute_d2e, _compute_scalar_traces
        d2e_ana = _compute_d2e(self.gost, self.dm)

        def get_dg(g, d):
            dg, _ = _compute_scalar_traces(g, d)
            return dg

        d2e_fd = _fd_d2_scalar(self.gost, self.dm, get_dg)
        np.testing.assert_allclose(d2e_ana, d2e_fd, atol=1e-5)

    def test_d2e_symmetry(self):
        """d2e satisfies d2e[A,B,x,y,g] = d2e[B,A,y,x,g]."""
        from pyscf.solvent._gostshyp_hess import _compute_d2e
        d2e = _compute_d2e(self.gost, self.dm)
        np.testing.assert_allclose(d2e, d2e.transpose(1, 0, 3, 2, 4), atol=1e-10)


class TestD2F(unittest.TestCase):
    """Tests for _compute_d2F — second derivative of Fhat (force) trace."""

    @classmethod
    def setUpClass(cls):
        from pyscf.solvent._gostshyp_hess import _compute_d2F, _compute_scalar_traces
        # N2/cc-pVDZ — clean system
        mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ',
                    unit='Angstrom', verbose=0)
        opts = {'cavity': 'vdw', 'pressure_mpa': 50_000,
                'npoints': 110, 'scaling_factor': 1.2}
        gost = GOSTSHYP(mol, options=opts)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)
        cls.mol = mol
        cls.gost = gost
        cls.dm = dm

    def test_d2F_vs_fd_n2(self):
        """d2F analytical matches fdiff on N2/cc-pVDZ."""
        from pyscf.solvent._gostshyp_hess import _compute_d2F, _compute_scalar_traces
        d2F_ana = _compute_d2F(self.gost, self.dm)

        def get_dF(g, d):
            _, dF = _compute_scalar_traces(g, d)
            return dF

        d2F_fd = _fd_d2_scalar(self.gost, self.dm, get_dF)
        np.testing.assert_allclose(d2F_ana, d2F_fd, atol=1e-5)

    def test_d2F_symmetry(self):
        """d2F satisfies d2F[A,B,x,y,g] = d2F[B,A,y,x,g]."""
        from pyscf.solvent._gostshyp_hess import _compute_d2F
        d2F = _compute_d2F(self.gost, self.dm)
        np.testing.assert_allclose(d2F, d2F.transpose(1, 0, 3, 2, 4), atol=1e-10)


REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'reference_data')

E2E_SYSTEMS = [
    ('h2_cc-pvdz', 1e-6),
    ('n2_cc-pvdz', 1e-6),
    ('coh2_cc-pvdz', 1e-5),
    ('h2o_cc-pvdz', 1e-5),
    ('co2_cc-pvdz', 1e-5),
    ('sf6_cc-pvdz', 1e-5),
]


@pytest.mark.parametrize('system_name,atol', E2E_SYSTEMS,
                         ids=[s[0] for s in E2E_SYSTEMS])
def test_e2e_hessian(system_name, atol):
    """End-to-end GOSTSHYP Hessian vs stored numerical reference."""
    from pyscf.solvent.test.reference_systems import make_mf

    ref_path = os.path.join(REFERENCE_DIR, f'hess_{system_name}.npy')
    if not os.path.exists(ref_path):
        pytest.skip(f'Reference not found: {ref_path}. '
                    f'Run: python -m pyscf.solvent.test.'
                    f'generate_gostshyp_hessian_references '
                    f'--systems {system_name}')

    hess_ref = np.load(ref_path)
    mf = make_mf(system_name)
    hess_ana = mf.Hessian().kernel()
    np.testing.assert_allclose(
        hess_ana, hess_ref, atol=atol,
        err_msg=f'{system_name}: analytical Hessian does not match reference')


# --- Parametrized gost.hess(dm) vs gost.hess_fd(dm) tests ---

HESS_VS_FD_SYSTEMS = [
    ('H2', 'H 0 0 0; H 0 0 1.4', 'cc-pVDZ', {'cavity': 'vdw'}),
    ('N2', 'N 0 0 0; N 0 0 1.098', 'cc-pVDZ', {'cavity': 'vdw'}),
    ('H2O', 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'cc-pVDZ',
     {'cavity': 'vdw'}),
]


@pytest.mark.parametrize('name,atom,basis,extra_opts', HESS_VS_FD_SYSTEMS,
                         ids=[s[0] for s in HESS_VS_FD_SYSTEMS])
def test_hess_vs_fd(name, atom, basis, extra_opts):
    """gost.hess(dm) matches gost.hess_fd(dm) at fixed density."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    opts = {'pressure_mpa': 50_000, 'npoints': 110, 'scaling_factor': 1.2}
    opts.update(extra_opts)
    gost = GOSTSHYP(mol, options=opts)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf = gostshyp_for_scf(mf, gost)
    mf.kernel()
    dm = mf.make_rdm1()
    gost.kernel(dm)
    H_ana = gost.hess(dm)
    H_fd = gost.hess_fd(dm, step=1e-4)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-5)


class TestBDotX(unittest.TestCase):
    """Tests for GOSTSHYP._B_dot_x — CPHF linear response kernel."""

    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
                    basis='cc-pVDZ', unit='Angstrom', verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2})
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = gostshyp_for_scf(mf, gost)
        mf.kernel()
        dm = mf.make_rdm1()
        gost.kernel(dm)

        cls.mol = mol
        cls.gost = gost
        cls.dm = dm

        rng = np.random.default_rng(42)
        nao = mol.nao_nr()
        dm1 = rng.standard_normal((nao, nao))
        cls.dm1 = 0.5 * (dm1 + dm1.T)

    def test_fd_h2o(self):
        """_B_dot_x matches finite differences of kernel() on H2O/cc-pVDZ."""
        gost, dm, dm1 = self.gost, self.dm, self.dm1
        v_ana = gost._B_dot_x(dm1)

        eps = 1e-5
        opts = {'cavity': gost.cavity, 'pressure_mpa': gost.pressure_mpa,
                'npoints': gost.npoints, 'scaling_factor': gost.scaling_factor}
        gost_p = GOSTSHYP(self.mol, options=opts)
        gost_p.kernel(dm + eps * dm1)
        gost_m = GOSTSHYP(self.mol, options=opts)
        gost_m.kernel(dm - eps * dm1)
        v_fd = (gost_p.v - gost_m.v) / (2 * eps)

        np.testing.assert_allclose(v_ana, v_fd, atol=1e-7)

    def test_symmetry(self):
        """_B_dot_x output is symmetric."""
        v = self.gost._B_dot_x(self.dm1)
        np.testing.assert_allclose(v, v.T, atol=1e-14)

    def test_direct_vs_cached(self):
        """Direct and cached modes give same result."""
        gost = self.gost
        dm1 = self.dm1

        # Force cached mode
        gost_cached = GOSTSHYP(self.mol, options={
            'cavity': 'vdw', 'pressure_mpa': 50_000,
            'npoints': 110, 'scaling_factor': 1.2, 'direct': False})
        gost_cached.kernel(self.dm)

        v_direct = gost._B_dot_x(dm1)
        v_cached = gost_cached._B_dot_x(dm1)
        np.testing.assert_allclose(v_direct, v_cached, atol=1e-12)

    def test_batched(self):
        """Batched (nset, nao, nao) input gives same result as individual calls."""
        gost = self.gost
        nao = self.mol.nao_nr()
        rng = np.random.default_rng(123)
        nset = 3
        dm1_batch = rng.standard_normal((nset, nao, nao))
        dm1_batch = 0.5 * (dm1_batch + dm1_batch.transpose(0, 2, 1))

        v_batch = gost._B_dot_x(dm1_batch)
        v_singles = np.array([gost._B_dot_x(dm1_batch[i]) for i in range(nset)])
        np.testing.assert_allclose(v_batch, v_singles, atol=1e-14)

    def test_shape_single(self):
        """Single dm1 input returns (nao, nao)."""
        v = self.gost._B_dot_x(self.dm1)
        nao = self.mol.nao_nr()
        self.assertEqual(v.shape, (nao, nao))

    def test_shape_batched(self):
        """Batched dm1 input returns (nset, nao, nao)."""
        nao = self.mol.nao_nr()
        dm1 = np.zeros((4, nao, nao))
        v = self.gost._B_dot_x(dm1)
        self.assertEqual(v.shape, (4, nao, nao))

    def test_kernel_not_called_raises(self):
        """Must call kernel() before _B_dot_x."""
        gost = GOSTSHYP(self.mol, options={'cavity': 'vdw'})
        with self.assertRaises((RuntimeError, AttributeError)):
            gost._B_dot_x(self.dm1)


if __name__ == '__main__':
    unittest.main()
