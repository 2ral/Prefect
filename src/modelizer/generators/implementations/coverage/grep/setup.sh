#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR
tar xf grep-3.11.tar.gz
mv grep-3.11 compiled_grep-3.11
cd compiled_grep-3.11
./configure CFLAGS="--coverage -g -O0 -Wall" CXXFLAGS="--coverage -g -O0 -Wall" LDFLAGS="--coverage"
make
make install
cd ..
