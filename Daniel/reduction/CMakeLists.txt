cmake_minimum_required(VERSION 3.17)
project(my_hpx_project CXX)
find_package(HPX REQUIRED)
find_package(benchmark REQUIRED)
add_executable(main reduction.cpp)
target_link_libraries(main HPX::hpx HPX::wrap_main HPX::iostreams_component HPX::partitioned_vector_component benchmark::benchmark)
