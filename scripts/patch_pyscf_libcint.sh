#!/usr/bin/env bash
# Patch conda-installed PySCF to use our custom libcint with int3c1e second derivatives.
# Run from the pyscf-forge root directory.
#
# Prerequisites:
#   - libcint built at libcint/build/libcint.dylib (see libcint/README)
#   - conda env "pyscf" active with pyscf 2.13.0 installed
#
# What this does:
#   1. Symlinks our libcint into conda pyscf's deps
#   2. Adds int3c1e_{ipip1,ip1ip2,ipvip1,ipip2} to _INTOR_FUNCTIONS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Find conda pyscf location
PYSCF_DIR=$(python -c "import pyscf, os; print(os.path.dirname(pyscf.__file__))")
echo "PySCF location: $PYSCF_DIR"

CONDA_LIBCINT="$PYSCF_DIR/lib/deps/lib/libcint.6.dylib"
LOCAL_LIBCINT="$REPO_ROOT/libcint/build/libcint.dylib"
MOLEINTOR="$PYSCF_DIR/gto/moleintor.py"

# Check prerequisites
if [ ! -f "$LOCAL_LIBCINT" ]; then
    echo "ERROR: Local libcint not found at $LOCAL_LIBCINT"
    echo "Build it first: cd libcint && mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

# 1. Symlink libcint
if [ -L "$CONDA_LIBCINT" ]; then
    echo "libcint symlink already exists, updating..."
fi
ln -sf "$LOCAL_LIBCINT" "$CONDA_LIBCINT"
echo "✓ Linked libcint: $CONDA_LIBCINT -> $LOCAL_LIBCINT"

# 2. Patch moleintor.py (idempotent)
if grep -q "int3c1e_ipip1" "$MOLEINTOR"; then
    echo "✓ moleintor.py already patched"
else
    sed -i '' "/    'int3c1e_ip1'               : (3, 3),/a\\
\\    'int3c1e_ipip1'             : (9, 9),\\
\\    'int3c1e_ip1ip2'            : (9, 9),\\
\\    'int3c1e_ipvip1'            : (9, 9),\\
\\    'int3c1e_ipip2'             : (9, 9),
" "$MOLEINTOR"
    echo "✓ Patched moleintor.py with int3c1e second-derivative integrals"
fi

# Verify
python -c "
from pyscf import gto
import numpy as np
mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', cart=True, verbose=0)
from pyscf.solvent.gostshyp import fakemol_for_gaussian
gmol = fakemol_for_gaussian(np.array([[0.,0.,0.5]]), np.array([1.0]))
sup = mol + gmol
s = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+1)
for name in ['int3c1e_ipip1','int3c1e_ip1ip2','int3c1e_ipvip1','int3c1e_ipip2']:
    sup.intor(name, shls_slice=s)
print('✓ All int3c1e second-derivative integrals verified')
"
