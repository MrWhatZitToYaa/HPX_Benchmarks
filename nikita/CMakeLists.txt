cmake_minimum_required(VERSION 3.17)
project(HPXTEST CXX)

#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_COMPILER g++)
#set(CMAKE_BUILD_TYPE Debug)
#set(BUILD_SHARED_LIBS true)

find_package(benchmark REQUIRED)
find_package(HPX REQUIRED)

add_executable(reduction.out reduction.cpp)
target_link_libraries(reduction.out HPX::hpx HPX::wrap_main HPX::iostreams_component HPX::partitioned_vector_component)
target_link_libraries(reduction.out benchmark::benchmark)

add_executable(transform.out transform.cpp)
target_link_libraries(transform.out HPX::hpx HPX::wrap_main HPX::iostreams_component HPX::partitioned_vector_component)
target_link_libraries(transform.out benchmark::benchmark)

add_executable(correct_runtime_test.out correct_runtime_test.cpp)
target_link_libraries(correct_runtime_test.out HPX::hpx HPX::wrap_main HPX::iostreams_component HPX::partitioned_vector_component)
target_link_libraries(correct_runtime_test.out benchmark::benchmark)

add_executable(custom_time_measurement.out custom_time_measurement.cpp)
target_link_libraries(custom_time_measurement.out HPX::hpx HPX::wrap_main HPX::iostreams_component HPX::partitioned_vector_component)