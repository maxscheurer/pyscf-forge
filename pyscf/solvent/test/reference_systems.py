"""
Reference system definitions for GOSTSHYP Hessian tests.

Each system is defined by its geometry, basis set, and GOSTSHYP options.
Reference Hessians (from fdiff of SCF gradient) are stored as .npy files
in the reference_data/ subdirectory, keyed by system name.

To regenerate references, run:
    python -m pyscf.solvent.test.generate_gostshyp_hessian_references --all
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GOSTSHYPOptions:
    """GOSTSHYP model options."""
    cavity: str = 'vdw'
    pressure_mpa: float = 50_000
    npoints: int = 110
    scaling_factor: float = 1.2
    r_ext: float = 0.4724  # only used for vdw/occ

    def to_dict(self):
        """Convert to dict for GOSTSHYP constructor."""
        d = {
            'cavity': self.cavity,
            'pressure_mpa': self.pressure_mpa,
            'npoints': self.npoints,
            'scaling_factor': self.scaling_factor,
        }
        if self.cavity == 'vdw/occ':
            d['r_ext'] = self.r_ext
        return d


@dataclass(frozen=True)
class ReferenceSystem:
    """A molecular system for Hessian reference testing."""
    molecule: str
    atom: str
    basis: str
    unit: str = 'Angstrom'
    gostshyp: GOSTSHYPOptions = field(default_factory=GOSTSHYPOptions)

    @property
    def key(self):
        """Dict key: molecule_basis (lowercase, hyphens)."""
        return f'{self.molecule}_{self.basis}'.lower().replace(' ', '')

    def make_mf(self):
        """Build converged SCF+GOSTSHYP for this system.

        Returns
        -------
        mf : SCF method (converged, with GOSTSHYP attached)
        """
        from pyscf import gto, scf
        from pyscf.solvent.gostshyp import GOSTSHYP

        mol = gto.M(atom=self.atom, basis=self.basis,
                    unit=self.unit, verbose=0)
        gost = GOSTSHYP(mol, options=self.gostshyp.to_dict())
        mf = scf.RHF(mol).GOSTSHYP(solvent_obj=gost)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-10
        mf.kernel()
        assert mf.converged, f'SCF did not converge for {self.name}'
        return mf


# Default GOSTSHYP options shared by all systems
_opts = GOSTSHYPOptions()

_SYSTEM_LIST = [
    ReferenceSystem(
        molecule='H2',
        atom='H 0 0 0; H 0 0 0.74',
        basis='cc-pVDZ',
    ),
    ReferenceSystem(
        molecule='N2',
        atom='N 0 0 0; N 0 0 1.098',
        basis='cc-pVDZ',
    ),
    ReferenceSystem(
        molecule='H2O',
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pVDZ',
    ),
    ReferenceSystem(
        molecule='COH2',
        atom='C 0 0 0; O 1.2 0 0; H -0.6 0.9 0; H -0.6 -0.9 0',
        basis='cc-pVDZ',
    ),
    ReferenceSystem(
        molecule='CO2',
        atom='O -1.16 0 0; C 0 0 0; O 1.16 0 0',
        basis='cc-pVDZ',
    ),
    ReferenceSystem(
        molecule='SF6',
        atom=('S  0.000  0.000  0.000; '
              'F  1.560  0.000  0.000; '
              'F -1.560  0.000  0.000; '
              'F  0.000  1.560  0.000; '
              'F  0.000 -1.560  0.000; '
              'F  0.000  0.000  1.560; '
              'F  0.000  0.000 -1.560'),
        basis='cc-pVDZ',
    ),
]

SYSTEMS = {s.key: s for s in _SYSTEM_LIST}


def make_mf(system_name):
    """Build converged SCF+GOSTSHYP for a reference system by name.

    Parameters
    ----------
    system_name : str
        Key into SYSTEMS dict.

    Returns
    -------
    mf : SCF method (converged, with GOSTSHYP attached)
    """
    return SYSTEMS[system_name].make_mf()
