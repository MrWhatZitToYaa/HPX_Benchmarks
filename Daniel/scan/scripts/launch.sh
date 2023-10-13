#!/usr/bin/env bash
#SBATCH --job-name=partitioned_vector_hpx
#SBATCH -p qdr
#SBATCH -N 1
###SBATCH --exclusive

spack load hpx
mpirun hostname
##--hpx:threads 2
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 32768 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 65536 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 131072 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 262144 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 524288 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 1048576 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 2097152 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 4194304 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 8388608 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 16777216 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 33554432 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 67108864 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 134217728 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 268435456 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 536870912 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 1073741824 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 2147483648 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 4294967296 --hpx:print-counter=/runtime{locality#*/total}/uptime
mpirun ./../build/main --hpx:ignore-batch-env --maxelems 8589934592 --hpx:print-counter=/runtime{locality#*/total}/uptime
