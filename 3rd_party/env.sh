#!/usr/bin/env bash

_this_script="$(readlink -f "${BASH_SOURCE[0]}")"
_third_party_prefix="$(cd "$(dirname "$_this_script")" && pwd)"

export THIRD_PARTY_PREFIX="$_third_party_prefix/install"
export PATH="$THIRD_PARTY_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$THIRD_PARTY_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$THIRD_PARTY_PREFIX/lib:$LIBRARY_PATH"
export CPATH="$THIRD_PARTY_PREFIX/include:$CPATH"
export CMAKE_PREFIX_PATH="$THIRD_PARTY_PREFIX:$CMAKE_PREFIX_PATH"
export PKG_CONFIG_PATH="$THIRD_PARTY_PREFIX/lib/pkgconfig:$THIRD_PARTY_PREFIX/share/pkgconfig:$PKG_CONFIG_PATH"


echo "Exported:"
echo "THIRD_PARTY_PREFIX=$THIRD_PARTY_PREFIX"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
