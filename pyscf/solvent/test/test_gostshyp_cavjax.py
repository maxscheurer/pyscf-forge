# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

import contextlib
import importlib.util
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import numpy as np

from pyscf import gto, scf
from pyscf.solvent import cavjax as cavjax_bridge
from pyscf.solvent.gostshyp import (
    GOSTSHYP,
    fakemol_for_gaussian,
    gostshyp_for_scf,
)


class _FakeBackend:
    instances: ClassVar = []

    def __init__(self, atomic_numbers, options):
        self.atomic_numbers = tuple(atomic_numbers)
        self.options = dict(options)
        self.n_points = 12
        self.n_shape_directions = 42
        self.build_calls = []
        self.response_calls = []
        type(self).instances.append(self)

    def build(self, positions):
        positions = np.asarray(positions)
        self.build_calls.append(positions.copy())
        directions = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0],
            [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1],
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        center = positions.mean(axis=0)
        return center + 3 * directions, np.ones(12), -directions

    def response(self, positions, points, areas, normals):
        self.response_calls.append(tuple(np.array(x, copy=True)
                                         for x in (points, areas, normals)))
        return np.zeros((len(positions), 3))


class TestOptionalBoundary(unittest.TestCase):
    def test_ordinary_import_does_not_import_cavjax(self):
        code = r'''
import builtins
import sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "cavjax" or name.startswith("cavjax."):
        raise AssertionError("ordinary GOSTSHYP imported CavJAX")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import importlib.util
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("pyscf.solvent.gostshyp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "cavjax" not in sys.modules
'''
        path = os.path.join(os.path.dirname(__file__), '..', 'gostshyp.py')
        subprocess.run([sys.executable, '-c', code, path], check=True,
                       env=dict(os.environ))

    def test_missing_dependency_has_installation_error(self):
        original = cavjax_bridge.importlib.import_module

        def import_module(name):
            if name == 'cavjax':
                raise ModuleNotFoundError('missing cavjax')
            return original(name)

        patcher = mock.patch.object(
            cavjax_bridge.importlib, 'import_module', side_effect=import_module)
        with patcher, self.assertRaisesRegex(
                ImportError, 'pyscf-forge.*cavjax'):
            cavjax_bridge.CavJAXBackend([1], {})

    def test_bridge_converts_normal_and_cotangent_sign_once(self):
        calls = {}

        class Model:
            n_points = 2
            n_shape_directions = 12

            def __init__(self, numbers, **kwargs):
                calls['init'] = (np.asarray(numbers).copy(), kwargs)

            def build(self, positions):
                calls['build'] = np.asarray(positions).copy()
                return SimpleNamespace(
                    points=np.array([[1., 0., 0.], [0., 1., 0.]]),
                    areas=np.array([2., 3.]),
                    normals=np.array([[1., 0., 0.], [0., 1., 0.]]))

            def response(self, positions, cotangents):
                calls['response'] = cotangents
                return np.ones((1, 3))

        fake_cavjax = SimpleNamespace(
            MolecularCavity=Model,
            SurfaceCotangent=lambda **kwargs: SimpleNamespace(**kwargs))
        fake_jax = SimpleNamespace(
            devices=lambda platform: [object()],
            default_device=lambda device: contextlib.nullcontext())
        fake_jnp = SimpleNamespace(asarray=np.asarray)

        def import_module(name):
            return {'cavjax': fake_cavjax, 'jax': fake_jax,
                    'jax.numpy': fake_jnp}[name]

        with mock.patch.object(cavjax_bridge.importlib, 'import_module',
                               side_effect=import_module):
            backend = cavjax_bridge.CavJAXBackend(
                [8], {'points_per_atom': 2, 'root_steps': 32})
        points, areas, inward = backend.build(np.zeros((1, 3)))
        response = backend.response(
            np.zeros((1, 3)), np.ones((2, 3)), np.ones(2),
            np.full((2, 3), 4.0))

        np.testing.assert_array_equal(inward, [[-1, 0, 0], [0, -1, 0]])
        np.testing.assert_array_equal(calls['response'].normals,
                                      np.full((2, 3), -4.0))
        np.testing.assert_array_equal(response, np.ones((1, 3)))
        self.assertTrue(points.flags.c_contiguous)
        self.assertTrue(areas.flags.c_contiguous)
        self.assertEqual(calls['init'][1]['root_steps'], 32)

    def test_bridge_rejects_invalid_surface_data(self):
        valid = SimpleNamespace(
            points=np.zeros((2, 3)), areas=np.ones(2),
            normals=np.array([[1., 0., 0.], [0., 1., 0.]]))

        class Model:
            n_points = 2
            n_shape_directions = 12
            surface = valid

            def __init__(self, numbers, **kwargs):
                pass

            def build(self, positions):
                return self.surface

        fake_cavjax = SimpleNamespace(MolecularCavity=Model)
        fake_jax = SimpleNamespace(
            devices=lambda platform: [object()],
            default_device=lambda device: contextlib.nullcontext())
        fake_jnp = SimpleNamespace(asarray=np.asarray)

        def import_module(name):
            return {'cavjax': fake_cavjax, 'jax': fake_jax,
                    'jax.numpy': fake_jnp}[name]

        with mock.patch.object(cavjax_bridge.importlib, 'import_module',
                               side_effect=import_module):
            backend = cavjax_bridge.CavJAXBackend([1], {})

        invalid = [
            (SimpleNamespace(points=np.zeros((3, 3)), areas=valid.areas,
                             normals=valid.normals), 'shape'),
            (SimpleNamespace(points=valid.points,
                             areas=np.array([1., np.nan]),
                             normals=valid.normals), 'non-finite'),
            (SimpleNamespace(points=valid.points, areas=np.array([1., 0.]),
                             normals=valid.normals), 'non-positive'),
            (SimpleNamespace(points=valid.points, areas=valid.areas,
                             normals=np.ones((2, 3))), 'non-unit'),
        ]
        for surface, message in invalid:
            with self.subTest(message=message):
                Model.surface = surface
                with self.assertRaisesRegex(ValueError, message):
                    backend.build(np.zeros((1, 3)))


class TestCavJAXLifecycle(unittest.TestCase):
    def setUp(self):
        _FakeBackend.instances.clear()
        self.patch = mock.patch(
            'pyscf.solvent.cavjax.CavJAXBackend', _FakeBackend)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def make_mol(self, distance=1.4, atoms='H; H'):
        labels = [item.strip() for item in atoms.split(';')]
        specification = '; '.join(
            f'{label} {i * distance} 0 0' for i, label in enumerate(labels))
        return gto.M(atom=specification, basis='sto-3g', verbose=0)

    def test_options_surface_and_reset_reuse(self):
        mol = self.make_mol()
        kwargs = {'points_per_atom': 17, 'root_steps': 40}
        gost = GOSTSHYP(mol, options={
            'cavity': 'cavjax', 'cavjax_kwargs': kwargs})
        backend = _FakeBackend.instances[0]
        self.assertEqual(backend.atomic_numbers, (1, 1))
        self.assertEqual(backend.options, kwargs)
        self.assertIsNone(gost.atom_idx)
        self.assertEqual(gost.n_gaussian, 12)
        self.assertGreaterEqual(gost._cavjax_build_t_wall, 0.0)

        kwargs['points_per_atom'] = 99
        moved = self.make_mol(distance=1.6)
        gost.e = 1.0
        gost.forces = np.ones(12)
        gost._grad_integral_t_wall = 2.0
        gost.reset(moved)
        self.assertIs(gost._cavjax_backend, backend)
        self.assertEqual(len(backend.build_calls), 2)
        self.assertEqual(backend.options['points_per_atom'], 17)
        self.assertIsNone(gost.e)
        self.assertIsNone(gost.forces)
        self.assertEqual(gost._grad_integral_t_wall, 0.0)

        changed = self.make_mol(atoms='H; Li')
        with self.assertRaisesRegex(ValueError, 'new GOSTSHYP object'):
            gost.reset(changed)
        self.assertIs(gost.mol, moved)

    def test_invalid_options_and_cached_mode_fail_before_backend(self):
        mol = self.make_mol()
        with self.assertRaisesRegex(TypeError, 'mapping'):
            GOSTSHYP(mol, options=[])
        with self.assertRaisesRegex(TypeError, 'cavjax_kwargs'):
            GOSTSHYP(mol, options={
                'cavity': 'cavjax', 'cavjax_kwargs': []})
        with self.assertRaisesRegex(ValueError, 'direct=True'):
            GOSTSHYP(mol, options={
                'cavity': 'cavjax', 'direct': False})
        self.assertEqual(_FakeBackend.instances, [])

    def test_cached_mode_cannot_be_enabled_after_construction(self):
        mol = self.make_mol()
        gost = GOSTSHYP(mol, options={'cavity': 'cavjax'})
        gost.direct = False
        with mock.patch.object(
                gost, '_kernel_cached',
                side_effect=AssertionError('cached operators must not be used')):
            with self.assertRaisesRegex(ValueError, 'direct=True'):
                gost.kernel(np.eye(mol.nao_nr()))


class TestCompactCotangents(unittest.TestCase):
    def test_matches_dense_rigid_surface_contraction(self):
        """Compact cotangents reproduce the established vdW direct gradient."""
        mol = gto.M(atom='H 1 0 0; F 2 0 0', basis='sto-3g', cart=True,
                    verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'vdw', 'npoints': 26, 'direct': True})
        dm = np.eye(mol.nao_nr())
        gost.kernel(dm)
        dense_reference = gost._grad_direct(dm)
        dareas, _, _ = gost._get_surface_derivatives()
        owner = gost.atom_idx.copy()
        rng = np.random.default_rng(12)
        dnormals = rng.normal(size=(3, 3, mol.natm, gost.n_gaussian))

        gmol_p = fakemol_for_gaussian(
            gost.grid_coords, gost.widths, l=1,
            coeffs=2.0 * gost.widths * gost.N_j)
        supermol_p = mol + gmol_p
        slices_p = (0, mol.nbas, 0, mol.nbas,
                    mol.nbas, mol.nbas + gmol_p.nbas)
        overlap3p = supermol_p.intor(
            'int3c1e', shls_slice=slices_p
        ).reshape(mol.nao_nr(), mol.nao_nr(), -1, 3)
        p_expectation = np.einsum(
            'ijgc,ij->gc', overlap3p, dm, optimize=True)
        expected_normal_cotangent = (
            -gost.pressure_au * gost.areas * gost.gtilde_expval
            / gost.forces**2)[:, None] * p_expectation
        normal_response = np.einsum(
            'gc,caAg->Aa', expected_normal_cotangent, dnormals,
            optimize=True)

        class Oracle:
            calls = 0

            def response(self, positions, points, areas, normals):
                self.calls += 1
                self.cotangents = (points.copy(), areas.copy(), normals.copy())
                result = np.einsum('Aag,g->Aa', dareas, areas,
                                   optimize=True)
                np.add.at(result, owner, points)
                result += np.einsum('gc,caAg->Aa', normals, dnormals,
                                    optimize=True)
                return result

        oracle = Oracle()
        gost.cavity = 'cavjax'
        gost.atom_idx = None
        gost._cavjax_backend = oracle
        gost._direct_gradient_chunk_size = mock.Mock(return_value=7)
        gost._scatter_gaussian_center_grad = mock.Mock(
            side_effect=AssertionError('owner scatter must not be used'))
        gost._normal_derivative_term = mock.Mock(
            side_effect=AssertionError('dense normal path must not be used'))
        compact = gost._grad_direct(dm)

        np.testing.assert_allclose(
            compact, dense_reference + normal_response, atol=2e-11)
        self.assertEqual(oracle.calls, 1)
        points, areas, normals = oracle.cotangents
        self.assertEqual(points.shape, (gost.n_gaussian, 3))
        self.assertEqual(areas.shape, (gost.n_gaussian,))
        self.assertEqual(normals.shape, (gost.n_gaussian, 3))
        np.testing.assert_allclose(normals, expected_normal_cotangent,
                                   atol=2e-12)
        self.assertGreater(gost._grad_integral_t_wall, 0.0)
        self.assertGreaterEqual(gost._grad_cotangent_t_wall, 0.0)
        self.assertGreaterEqual(gost._grad_cavjax_vjp_t_wall, 0.0)


@unittest.skipUnless(importlib.util.find_spec('cavjax'),
                     'CavJAX is not installed')
class TestInstalledCavJAX(unittest.TestCase):
    def setUp(self):
        if os.environ.get('JAX_ENABLE_X64') not in ('1', 'true', 'True'):
            self.skipTest('JAX float64 is not enabled')

    def test_real_surface_and_independent_response_components(self):
        mol = gto.M(atom='O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1',
                    unit='Bohr', basis='sto-3g', verbose=0)
        backend = cavjax_bridge.CavJAXBackend(
            mol.atom_charges(), {'points_per_atom': 6,
                                 'target_shape_directions': 12,
                                 'root_steps': 48})
        positions = mol.atom_coords()
        points, areas, inward = backend.build(positions)
        self.assertEqual(points.shape, (backend.n_points, 3))
        self.assertTrue(np.all(areas > 0))
        np.testing.assert_allclose(np.linalg.norm(inward, axis=1), 1.0,
                                   atol=1e-10)

        rng = np.random.default_rng(7)
        direction = rng.normal(size=positions.shape)
        h = 2e-5
        plus = backend.build(positions + h * direction)
        minus = backend.build(positions - h * direction)
        for index, cotangent in enumerate((
                rng.normal(size=points.shape), rng.normal(size=areas.shape),
                rng.normal(size=inward.shape))):
            cots = [np.zeros_like(points), np.zeros_like(areas),
                    np.zeros_like(inward)]
            cots[index] = cotangent
            response = backend.response(positions, *cots)
            finite_difference = np.vdot(cotangent,
                                         (plus[index] - minus[index]) / (2*h))
            self.assertAlmostEqual(np.vdot(response, direction),
                                   finite_difference, places=6)

    def _check_scf_directional_gradient(self, atom):
        mol = gto.M(atom=atom, unit='Bohr', basis='sto-3g', verbose=0)
        gost = GOSTSHYP(mol, options={
            'cavity': 'cavjax', 'pressure_mpa': 1000,
            'cavjax_kwargs': {'points_per_atom': 4,
                              'target_shape_directions': 12,
                              'root_steps': 32}})
        mf = gostshyp_for_scf(scf.RHF(mol), gost)
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)
        gradient = mf.Gradients().kernel()
        reference_energy = mf.e_tot
        np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-7)
        direction = np.arange(mol.natm * 3, dtype=float).reshape(mol.natm, 3)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        reference = mol.atom_coords()
        scanner = mf.as_scanner()
        translated = mol.set_geom_(reference + np.array([0.2, -0.3, 0.1]),
                                   unit='Bohr', inplace=False)
        np.testing.assert_allclose(scanner(translated), reference_energy,
                                   atol=1e-9)
        h = 3e-4
        e_plus = scanner(mol.set_geom_(reference + h * direction,
                                       unit='Bohr', inplace=False))
        e_minus = scanner(mol.set_geom_(reference - h * direction,
                                        unit='Bohr', inplace=False))
        self.assertAlmostEqual(np.vdot(gradient, direction),
                               (e_plus - e_minus) / (2*h), places=5)

    def test_closed_shell_molecule_scf_gradient(self):
        self._check_scf_directional_gradient('H 0 0 0; H 1.4 0 0')

    def test_intermolecular_cluster_scf_gradient(self):
        self._check_scf_directional_gradient(
            'H 0 0 0; H 1.4 0 0; H 0 0 5; H 1.4 0 5')


if __name__ == '__main__':
    unittest.main()
