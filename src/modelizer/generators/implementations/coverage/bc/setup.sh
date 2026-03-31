#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR
tar xf bc-1.08.1.tar.xz
mv bc-1.08.1 compiled_bc-1.08.1
cd compiled_bc-1.08.1
./configure CFLAGS="--coverage -g -O0 -Wall" CXXFLAGS="--coverage -g -O0 -Wall" LDFLAGS="--coverage"
make
make install
cd ..
