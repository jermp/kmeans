# KMeans for byte vectors

A small, header-only, parallel implementation of kmeans clustering for arbitrary-long byte vectors.

The code is inspired by the [dkm](https://github.com/genbattle/dkm) library.


Compiling the code
------------------

The code is tested on Linux with `gcc` and on MacOS with `clang`.
To build the code, [`CMake`](https://cmake.org/) is required.

First clone the repository with

    git clone --recursive https://github.com/jermp/kmeans.git

If you forgot `--recursive` when cloning, do

    git submodule update --init --recursive

before compiling.

To compile the code for a release environment (see file `CMakeLists.txt` for the used compilation flags), it is sufficient to do the following, within the parent `kmeans` directory:

    mkdir build
    cd build
    cmake ..
    make -j

For a testing environment, use the following instead:

    mkdir debug_build
    cd debug_build
    cmake .. -D CMAKE_BUILD_TYPE=Debug -D KMEANS_USE_SANITIZERS=On
    make -j


Dependencies
------------

Parallelism is achieved via OpenMP.
To install it, do

	brew install libomp

on a Mac, or

	sudo apt install libomp

on Linux.
