# HPX_Benchmarks
Here we try out several different benchmarks, that run on a university cluster (Heidelberg) to evaluate the performance of the HPX runtime.

## Improvements after the Discussion on October 19, 2023

The following improvements should be implemented for the respective programs:

### For "Transform", "Reduction" and "Scan":

1. **Adjusted performance measurements with cache warm-up (not measuring the entire program runtime):**<br><br>
Has been implemented.

2. **Execution policy “par_simd”:** <br><br>
Not provided with the current HPX installation. The documentation states that a new datapar backend SVE must be installed to execute par_simd.<br><br>
See the documentation:<br><br>
https://hpx-docs.stellar-group.org/branches/master/html/releases/whats_new_1_9_0.html?highlight=simd <br><br>
https://github.com/STEllAR-GROUP/hpx/blob/master/cmake/HPX_SetupDatapar.cmake

### Additionally, for "Reduction" and "Scan":

1. **Determine how the data of a partitioned_vector is divided (large blocks or many small ones):** <br><br>
The documentation states, that it is a "Dynamic segmented contiguous array".<br><br>
See documentation:<br><br>
https://hpx-docs.stellar-group.org/branches/master/html/manual/writing_distributed_hpx_applications.html



### Additionally, for "Scan":

1. **Execute the second *inclusive_scan()* with the respective starting value from *sums_per_locality* vector (eliminating the need of the last Transform in the old Scan implementation):** <br><br>
This has been implemented and resulted in a significant performance improvement with multiple nodes.


2. **Use the *sums_per_locality* vector as a shared vector instead of a partitioned_vector, so that all localities have access to the entire vector:**<br><br>
We did not find explicit information about this in the documentation. Therefore, we conducted various experiments based on our own considerations, but without success.


### The following adjustment, in addition to the requested ones, was made: 
1. We had to use *hpx::distributed::barrier* instead of *hpx::distributed::latch* for the “Reduction” and “Scan” due to the revised performance measurement process (execution in a loop).


### General Information:
Performance measurements are performed on the qdr-partition.

We did not repeat the performance tests on the rome-partition, because there are currently problems on this partition, which lead to program terminations.




