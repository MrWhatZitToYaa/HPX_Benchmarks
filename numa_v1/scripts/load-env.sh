#!/bin/bash

echo "Setting up environment variables"
spack load gcc@13.1.0
spack load intel-oneapi-compilers@2023.1.0
spack load intel-tbb@2021.9.0
spack load benchmark@1.8.3
spack load cmake@3.26.3%gcc@=13.1.0
export CC=icx
export CXX=icpx


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/asc/pub/spack/opt/spack/linux-centos7-x86_64_v2/gcc-11.3.0/gcc-12.1.0-xz357upi6s7gnw23fmjob2oeqvawnp4q/lib64
