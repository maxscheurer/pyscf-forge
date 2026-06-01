"""
Reference system definitions for GOSTSHYP Hessian tests.

Each system is defined by its geometry, basis set, and GOSTSHYP options.
Reference Hessians (from fdiff of SCF gradient) are stored as .npy files
in the reference_data/ subdirectory, keyed by system name.

To regenerate references, run:
    python -m pyscf.solvent.test.generate_references [--systems sys1 sys2 ...]
"""

SYSTEMS = {
    'h2_cc-pvdz': {
        'atom': 'H 0 0 0; H 0 0 0.74',
        'basis': 'cc-pVDZ',
        'unit': 'Angstrom',
        'gostshyp': {
            'cavity': 'vdw',
            'pressure_mpa': 50_000,
            'npoints': 110,
            'scaling_factor': 1.2,
        },
    },
    'n2_cc-pvdz': {
        'atom': 'N 0 0 0; N 0 0 1.098',
        'basis': 'cc-pVDZ',
        'unit': 'Angstrom',
        'gostshyp': {
            'cavity': 'vdw',
            'pressure_mpa': 50_000,
            'npoints': 110,
            'scaling_factor': 1.2,
        },
    },
    'h2o_cc-pvdz': {
        'atom': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        'basis': 'cc-pVDZ',
        'unit': 'Angstrom',
        'gostshyp': {
            'cavity': 'vdw',
            'pressure_mpa': 50_000,
            'npoints': 110,
            'scaling_factor': 1.2,
        },
    },
    'coh2_cc-pvdz': {
        'atom': 'C 0 0 0; O 1.2 0 0; H -0.6 0.9 0; H -0.6 -0.9 0',
        'basis': 'cc-pVDZ',
        'unit': 'Angstrom',
        'gostshyp': {
            'cavity': 'vdw',
            'pressure_mpa': 50_000,
            'npoints': 110,
            'scaling_factor': 1.2,
        },
    },
    'co2_cc-pvdz': {
        'atom': 'O -1.16 0 0; C 0 0 0; O 1.16 0 0',
        'basis': 'cc-pVDZ',
        'unit': 'Angstrom',
        'gostshyp': {
            'cavity': 'vdw',
            'pressure_mpa': 50_000,
            'npoints': 110,
            'scaling_factor': 1.2,
        },
    },
    'sf6_cc-pvdz': {
        'atom': ('S  0.000  0.000  0.000; '
                 'F  1.560  0.000  0.000; '
                 'F -1.560  0.000  0.000; '
                 'F  0.000  1.560  0.000; '
                 'F  0.000 -1.560  0.000; '
                 'F  0.000  0.000  1.560; '
                 'F  0.000  0.000 -1.560'),
        'basis': 'cc-pVDZ',
        'unit': 'Angstrom',
        'gostshyp': {
            'cavity': 'vdw',
            'pressure_mpa': 50_000,
            'npoints': 110,
            'scaling_factor': 1.2,
        },
    },
}


def make_mf(system_name):
    """Build converged SCF+GOSTSHYP for a reference system.

    Parameters
    ----------
    system_name : str
        Key into SYSTEMS dict.

    Returns
    -------
    mf : SCF method (converged, with GOSTSHYP attached)
    """
    from pyscf import gto, scf
    from pyscf.solvent.gostshyp import GOSTSHYP

    sys = SYSTEMS[system_name]
    mol = gto.M(atom=sys['atom'], basis=sys['basis'],
                unit=sys['unit'], verbose=0)
    gost = GOSTSHYP(mol, options=sys['gostshyp'])
    mf = scf.RHF(mol).GOSTSHYP(solvent_obj=gost)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged, f'SCF did not converge for {system_name}'
    return mf
