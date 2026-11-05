#!/usr/bin/env bash
set -e

root_dir=$(pwd)

version=$(cat meshoptimizer_tag)

mkdir -p build
cd build

if [ ! -d "meshoptimizer" ]
then
	git clone -b "$version" https://github.com/zeux/meshoptimizer.git -c advice.detachedHead=false
fi

echo "Compiling meshoptimizer ($version)..."

cd meshoptimizer
make -s config=release build/release/libmeshoptimizer.a

mv build/release/libmeshoptimizer.a "$root_dir"/libmeshoptimizer.a

cd "$root_dir"
