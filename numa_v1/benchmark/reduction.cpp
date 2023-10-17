#include <benchmark/benchmark.h>
#include <vector>
#include <iostream>

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/partitioner.h>

#include "arenaV3.hpp"
#include "numa_adaptor.hpp"

using ValueType = float;
using ContainerType = std::vector<float, numa::no_init_allocator<float>>;
using namespace oneapi;
using Partitioner = oneapi::tbb::auto_partitioner;
static constexpr int gs = 4194304;
static constexpr int numa_nodes = 8;
static constexpr int thrds_per_node = 4;


static void Args(benchmark::internal::Benchmark* b) {
  const int64_t lowerLimit = 15;
  const int64_t upperLimit = 33;

  for (auto x = lowerLimit; x <= upperLimit; ++x) {
        b->Args({int64_t{1} << x});
  }
}

void setCustomCounter(benchmark::State& state, std::string name) {
  state.counters["Elements"] = state.range(0);
  state.counters["Bytes"] = state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

static void benchReduceOmpNoInit(benchmark::State& state){
    ContainerType X(state.range(0));
    auto places = omp_get_num_places();
    size_t partSize = X.size() / places;
    #pragma omp parallel num_threads(numa_nodes) proc_bind(spread)
    {
        size_t part = omp_get_place_num();
        std::uninitialized_fill(std::execution::par_unseq, X.begin() + part * partSize, X.begin() + (part + 1) * partSize, ValueType{1});
    }

    ValueType sum;
    for (auto _ : state){
        sum = 0;
        #pragma omp parallel for simd reduction(+ : sum)
        for(size_t i = 0; i < X.size(); i++){
            sum += X[i];
        }
        benchmark::DoNotOptimize(&sum);
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "ReduceOmpNoInit");
}

static void benchReduceOmpNoInit2(benchmark::State& state){
    ContainerType X(state.range(0));
    auto places = omp_get_num_places();
    
    #pragma omp parallel for 
    for(size_t i = 0; i < X.size(); i++){
        new(&X[i]) ContainerType::value_type{1};
    }

    ValueType sum;
    for (auto _ : state){
        sum = 0;
        #pragma omp parallel for simd reduction(+ : sum)
        for (size_t i = 0; i < X.size(); i++){
            sum += X[i];
        }
        benchmark::DoNotOptimize(&sum);
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "ReduceOmpNoInit2");
}

static void benchReduceOmpNestingNoInit(benchmark::State& state){
    omp_set_max_active_levels(2);                                                                   // enable nested parallelism
    ContainerType X(state.range(0));
    auto places = omp_get_num_places();
    size_t partSize = X.size() / places;
    
    #pragma omp parallel num_threads(numa_nodes) proc_bind(spread)
    {
        size_t part = omp_get_place_num();
        std::uninitialized_fill(std::execution::par_unseq, X.begin() + part * partSize, X.begin() + (part + 1) * partSize, ValueType{1});
    }


    ValueType sum;
    for (auto _ : state){
        sum = 0;
        #pragma omp parallel num_threads(numa_nodes) proc_bind(spread)
        {
            size_t part = omp_get_place_num();
            size_t start = part * partSize;
            size_t end = (part + 1) * partSize;
            #pragma omp parallel for simd num_threads(thrds_per_node) proc_bind(master) reduction(+ : sum)
            for(size_t i = start; i < end; i++){
                sum += X[i];
            }
        }
        benchmark::DoNotOptimize(&sum);
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "ReduceOmpNestingNoInit");
}

static void benchReduceOmpNestingNoInit2(benchmark::State& state){
    omp_set_max_active_levels(2);
    ContainerType X(state.range(0));
    auto places = omp_get_num_places();
    size_t partSize = X.size() / places;
    
    #pragma omp parallel num_threads(numa_nodes) proc_bind(spread)
    {
        size_t part = omp_get_place_num();
        size_t start = part * partSize;
        size_t end = (part + 1) * partSize;
        #pragma omp parallel for simd num_threads(thrds_per_node) proc_bind(master)
        for(size_t i = start; i < X.size(); i++){
            new(&X[i]) ContainerType::value_type{1};
        }

    }

    ValueType sum;
    for (auto _ : state){
        sum = 0;
        #pragma omp parallel num_threads(numa_nodes) proc_bind(spread)
        {
            size_t part = omp_get_place_num();
            size_t start = part * partSize;
            size_t end = (part + 1) * partSize;
            #pragma omp parallel for simd num_threads(thrds_per_node) proc_bind(master) reduction(+ : sum)
            for(size_t i = start; i < end; i++){
                sum += X[i];
            }
        }
        benchmark::DoNotOptimize(&sum);
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "ReduceOmpNestingNoInit2");
}

static void benchReduceTbbNoInit(benchmark::State& state){
    numa::ArenaMgtTBBV3 arenas;                    
    numa_adaptor<ValueType, ContainerType> X(state.range(0), 1, arenas.get_nodes());
    std::vector<ValueType> node_sums(arenas.get_nodes());
    ValueType result;
    Partitioner part;

    for (auto _ : state){
        result = 0;
        arenas.execute([&] (const int i) {
            node_sums[i] = tbb::parallel_reduce(tbb::blocked_range<size_t>(X.get_range(i).first, X.get_range(i).second, gs), ValueType{0}, 
                                            [&] (const tbb::blocked_range<size_t> r, ValueType ret) -> ValueType {
                #pragma omp simd reduction(+ : ret)
                for (auto j = r.begin(); j < r.end(); j++) {
                    ret += X[j];
                }
                return ret;
            }, std::plus<>{}, part);
        });

        for (auto sum : node_sums) result += sum;
        benchmark::DoNotOptimize(&result);
        benchmark::ClobberMemory();
    }
    
    setCustomCounter(state, "ReduceTbbNoInitV7");
}

static void benchReduceTbbNoInit2(benchmark::State& state){
    numa::ArenaMgtTBBV3 arenas;                    
    numa_adaptor<ValueType, ContainerType> X(state.range(0), 1, arenas);
    std::vector<ValueType> node_sums(arenas.get_nodes());
    ValueType result;

    Partitioner part;

    for (auto _ : state){
        result = 0;
        arenas.execute([&] (const int i) {
            node_sums[i] = tbb::parallel_reduce(tbb::blocked_range<size_t>(X.get_range(i).first, X.get_range(i).second, gs), ValueType{0}, 
                                            [&] (const tbb::blocked_range<size_t> r, ValueType ret) -> ValueType {
                #pragma omp simd reduction(+ : ret)
                for (auto j = r.begin(); j < r.end(); j++) {
                    ret += X[j];
                }
                return ret;
            }, std::plus<>{}, part);
        });

        for (auto sum : node_sums) result += sum;    
        benchmark::DoNotOptimize(&result);
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "ReduceTbbNoInitV7");
}

BENCHMARK(benchReduceOmpNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceOmpNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceOmpNestingNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceOmpNestingNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchReduceTbbNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();