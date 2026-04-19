#!/usr/bin/env bash
set -e

root_dir=$(pwd)
mkdir -p build
cd build

if [ ! -d "meshoptimizer" ]
then
	git clone https://github.com/zeux/meshoptimizer.git --revision b22872835dbabc56a6e4a366ea9917f62b7daf1a -c advice.detachedHead=false
fi

echo "Compiling meshoptimizer..."

cd meshoptimizer
make -s config=release build/release/libmeshoptimizer.a

mv build/release/libmeshoptimizer.a "$root_dir"/libmeshoptimizer.a

cd "$root_dir"
