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


Examples
--------

The tool `tools/cluster` can be used to cluster a collection of byte vectors.
We assume the input collection `vectors.bin` is a binary file where: the
first 8 bytes encode the number of bytes per vector, say `p`; the next 8 bytes encode the number
of vectors in the collection, say `n`; we have then `p` bytes for vector (a total of `np` bytes).

	./cluster -i vectors.bin -k 16 -d 0.0 -s 13 > labels.txt

    ./cluster -i vectors.bin -m 7 -d 0.001 -s 13 --mse 500 --mcs 10 > labels.txt

    ./cluster -i vectors.bin -m 7 -d 0.001 -s 13 --mse 50 --mcs 1 > labels.txt