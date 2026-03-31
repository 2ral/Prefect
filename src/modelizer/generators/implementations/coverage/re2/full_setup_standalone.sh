#!/bin/bash

set -e  # Stop on error
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Installing core build tools..."
apt update
apt install -y cmake git build-essential pkg-config unzip

# === Install Google Test ===
echo "Installing Google Test..."
rm -rf ~/googletest
git clone https://github.com/google/googletest.git ~/googletest
cd ~/googletest
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
cd ~

# === Install Google Benchmark ===
echo "Installing Google Benchmark..."
rm -rf ~/benchmark
git clone https://github.com/google/benchmark.git ~/benchmark
cd ~/benchmark
git clone https://github.com/google/googletest.git  # Needed as submodule
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
cd ~

# === Install Abseil ===
echo "Installing Abseil..."
rm -rf ~/abseil-cpp
git clone https://github.com/abseil/abseil-cpp.git ~/abseil-cpp
cd ~/abseil-cpp

cmake -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DABSL_ENABLE_INSTALL=ON \
    -DABSL_USE_EXTERNAL_GOOGLETEST=ON \
    -DABSL_PROPAGATE_CXX_STD=ON


cmake --build build
cmake --install build

# Update pkg-config environment
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
source ~/.bashrc

pkg-config --exists absl_base && echo "✅ Abseil successfully installed"

cd $SCRIPT_DIR

# === Install RE2 ===
echo "Installing RE2..."
rm -f /usr/local/lib/libre2.*
rm -rf /usr/local/include/re2
rm -rf re2-main
rm -rf ~/re-main
unzip re2-main.zip
cd re2-main
make clean
make CXXFLAGS="--coverage -g -O0 -Wall -std=c++17 -DABSL_MIN_LOG_LEVEL=4 -DRE2_NO_THREADS" \
    LDFLAGS="--coverage"
make install
echo "✅ All components installed successfully!"

# === Compiling wrapper ===
cd $SCRIPT_DIR
g++ wrapper.cpp -std=c++17 -O0 -g --coverage -I/usr/local/include -lre2 -pthread -o wrapper

echo "/usr/local/lib" | tee /etc/ld.so.conf.d/re2.conf
ldconfig

echo "✅ Compilation complete: ./wrapper"
