cmake_minimum_required( VERSION 3.14 )
project(NUMA CXX)

find_package( Threads REQUIRED )
find_package( TBB REQUIRED )
find_package( benchmark REQUIRED )
find_package( OpenMP REQUIRED COMPONENTS CXX)

function( rome_build targetname )
	target_compile_features( ${targetname} PRIVATE cxx_std_20 )
	target_compile_options( ${targetname} PRIVATE -march=core-avx2 -mtune=core-avx2 -qopenmp-simd -O3 -mfma) 
	target_include_directories( ${targetname} PRIVATE include )
	target_link_libraries( ${targetname} PRIVATE benchmark::benchmark TBB::tbb Threads::Threads OpenMP::OpenMP_CXX )
endfunction()

function( qdr_build targetname )
	target_compile_features( ${targetname} PRIVATE cxx_std_20 )
	target_compile_options( ${targetname} PRIVATE -march=core-avx-i -mtune=core-avx-i -qopenmp-simd -O3 ) 
	target_include_directories( ${targetname} PRIVATE include )
	target_link_libraries( ${targetname} PRIVATE benchmark::benchmark TBB::tbb Threads::Threads OpenMP::OpenMP_CXX )
endfunction()

add_executable( reduction-benchmark_rome benchmark/reduction.cpp )
rome_build( reduction-benchmark_rome )

add_executable( reduction-benchmark_qdr benchmark/reduction.cpp )
qdr_build( reduction-benchmark_qdr )

add_executable( transform-benchmark_rome benchmark/transform.cpp )
rome_build( transform-benchmark_rome )

add_executable( transform-benchmark_qdr benchmark/transform.cpp )
qdr_build( transform-benchmark_qdr )


