#!/usr/bin/env bash

#SBATCH -p qdr
#SBATCH -N 2
#SBATCH -n 2

#SBATCH --job-name=transform_ROME
#SBATCH -o outROME_transfrom.txt

REPS=50

spack load hpx
mpirun hostname
#mpirun ./../build/transform --hpx:ignore-batch-env --hpx:threads 1 \
#							 --benchmark_time_unit=us --benchmark_out_format=csv --benchmark_out=transform_rome.csv \
#                            --benchmark_report_aggregates_only=true --benchmark_repetitions=$REPS \
#                            --benchmark_min_time=0.001 --benchmark_min_warmup_time=0.01
#
mpirun ./../build/correct_runtime_test  --hpx:ignore-batch-env --hpx:threads 1 \
										--benchmark_time_unit=us --benchmark_out_format=csv --benchmark_out=transform_rome.csv \
                            			--benchmark_report_aggregates_only=true --benchmark_repetitions=$REPS \
                            			--benchmark_min_time=0.001 --benchmark_min_warmup_time=0.01

