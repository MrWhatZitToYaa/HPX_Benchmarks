#!/bin/bash

echo "Setting up environment variables"
spack load gcc@12.1.0
spack load intel-oneapi-compilers@2023.1.0
spack load intel-oneapi-tbb@2021.7.1%oneapi
spack load benchmark@1.7.1
spack load cmake@3.24.3%oneapi
export CC=icx
export CXX=icpx


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/asc/pub/spack/opt/spack/linux-centos7-x86_64_v2/gcc-11.3.0/gcc-12.1.0-xz357upi6s7gnw23fmjob2oeqvawnp4q/lib64
