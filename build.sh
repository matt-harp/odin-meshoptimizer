#!/usr/bin/env bash
set -e

root_dir=$(pwd)
mkdir -p build
cd build

if [ ! -d "meshoptimizer" ]
then
	git clone https://github.com/zeux/meshoptimizer.git --branch v1.3 -c advice.detachedHead=false
fi

echo "Compiling meshoptimizer..."

cd meshoptimizer
make -s config=release build/release/libmeshoptimizer.a

mv build/release/libmeshoptimizer.a "$root_dir"/libmeshoptimizer.a

cd "$root_dir"
