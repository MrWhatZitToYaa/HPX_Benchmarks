#include <benchmark/benchmark.h>
#include <vector>

#include "allocator_adaptor.hpp"
#include "numa_adaptor.hpp"
#include "arenaV3.hpp"

using ValueType = float;
using ContainerType = std::vector<ValueType, numa::no_init_allocator<ValueType>>;
using namespace oneapi;
using Partitioner = tbb::static_partitioner;
static constexpr int numa_nodes = 4;
static constexpr int thrds_per_node = 32;

static void Args(benchmark::internal::Benchmark* b) {
  const int64_t lowerLimit = 15;
  const int64_t upperLimit = 33;

  for (auto x = lowerLimit; x <= upperLimit; ++x) {
    b->Args({int64_t{1} << x});
  }
}

void setCustomCounter(benchmark::State& state, std::string name) {
  state.counters["Elements"] = state.range(0);
  state.counters["Bytes"] = 3 * state.range(0) * sizeof(ValueType);
  state.SetLabel(name);
}

static void benchTransformOmpNoInit(benchmark::State& state){
    ContainerType X(state.range(0));
    ContainerType Y(state.range(0));
    size_t partSize = X.size() / numa_nodes;

    #pragma omp parallel proc_bind(spread) num_threads(numa_nodes)
    {
        auto place = omp_get_place_num();
        std::uninitialized_fill(std::execution::par_unseq, X.begin() + place * partSize, X.begin() + (place + 1) * partSize, ValueType{1});
        std::uninitialized_fill(std::execution::par_unseq, Y.begin() + place * partSize, Y.begin() + (place + 1) * partSize, ValueType{1});
    }
    constexpr ValueType alpha = 2;

    for (auto _ : state){
        #pragma omp parallel for 
        for (size_t i = 0; i < X.size(); i++){
            Y[i] = alpha * X[i] + Y[i];
        }
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "TransformOmpNoInit");
}

static void benchTransformOmpNoInit2(benchmark::State& state){
    ContainerType X(state.range(0));
    ContainerType Y(state.range(0));
    size_t partSize = X.size() / numa_nodes;

    #pragma omp parallel for 
    for (size_t i = 0; i < X.size(); i++){
        new(&X[i]) ContainerType::value_type{1};
        new(&Y[i]) ContainerType::value_type{2};
    }

    constexpr ValueType alpha = 2;

    for (auto _ : state){
        #pragma omp parallel for 
        for (size_t i = 0; i < X.size(); i++){
            Y[i] = alpha * X[i] + Y[i];
        }
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "TransformOmpNoInit2");
}

static void benchTransformOmpNestingNoInit(benchmark::State& state){
    omp_set_max_active_levels(2);
    ContainerType X(state.range(0));
    ContainerType Y(state.range(0));
    size_t partSize = X.size() / numa_nodes;

    #pragma omp parallel proc_bind(spread) num_threads(numa_nodes)
    {
        auto place = omp_get_place_num();
        std::uninitialized_fill(std::execution::par_unseq, X.begin() + place * partSize, X.begin() + (place + 1) * partSize, ValueType{1});
        std::uninitialized_fill(std::execution::par_unseq, Y.begin() + place * partSize, Y.begin() + (place + 1) * partSize, ValueType{1});
    }
    
    constexpr ValueType alpha = 2;

    for (auto _ : state){
        #pragma omp parallel proc_bind(spread) num_threads(numa_nodes)
        {
            auto part = omp_get_place_num();
            auto start = part * partSize;
            auto end = (part + 1) * partSize;
            
            #pragma omp parallel for proc_bind(master) num_threads(thrds_per_node) 
            for (size_t i = start; i < end; i++){
                Y[i] = alpha * X[i] + Y[i];
            }
        }
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "TransformOmpNestingNoInit");
}

static void benchTransformOmpNestingNoInit2(benchmark::State& state){
    ContainerType X(state.range(0));
    ContainerType Y(state.range(0));
    size_t partSize = X.size() / numa_nodes;

    #pragma omp parallel proc_bind(spread) num_threads(numa_nodes)
    {
        auto part = omp_get_place_num();
        auto start = part * partSize;
        auto end = (part + 1) * partSize;
        #pragma omp parallel for proc_bind(master) num_threads(thrds_per_node) 
        for (size_t i = start; i < end; i++){
            new(&X[i]) ContainerType::value_type{1};
            new(&Y[i]) ContainerType::value_type{2};
        }
    }

    constexpr ValueType alpha = 2;

    for (auto _ : state){
        #pragma omp parallel proc_bind(spread) num_threads(numa_nodes)
        {
            auto part = omp_get_place_num();
            auto start = part * partSize;
            auto end = (part + 1) * partSize;
            #pragma omp parallel for proc_bind(master) num_threads(thrds_per_node)
            for (size_t i = start; i < end; i++){
                Y[i] = alpha * X[i] + Y[i];
            }
        }
        benchmark::ClobberMemory();
    }

    setCustomCounter(state, "TransformOmpNestingNoInit2");
}

static void benchTransformTbbNoInit(benchmark::State& state) {
    numa::ArenaMgtTBBV3 arena;
    numa_adaptor<ValueType, ContainerType> X(state.range(0), 1, arena.get_nodes());
    numa_adaptor<ValueType, ContainerType> Y(state.range(0), 1, arena.get_nodes());

    constexpr ValueType alpha = 2;

    Partitioner part;

    for (auto _ : state) {
        arena.execute([&] (const int i) {
            tbb::parallel_for(tbb::blocked_range<size_t>(X.get_range(i).first, X.get_range(i).second), [&] (const tbb::blocked_range<size_t> r) {
                #pragma omp simd
                for (auto j = r.begin(); j < r.end(); j++) {
                    Y[j] = alpha * X[j] + Y[j];
                }
            }, part);
        });
    }
}

static void benchTransformTbbNoInit2(benchmark::State& state) {
    numa::ArenaMgtTBBV3 arena;
    numa_adaptor<ValueType, ContainerType> X(state.range(0), 1, arena);
    numa_adaptor<ValueType, ContainerType> Y(state.range(0), 1, arena);

    constexpr ValueType alpha = 2;

    Partitioner part;

    for (auto _ : state) {
        arena.execute([&, alpha] (const int i) {
            tbb::parallel_for(tbb::blocked_range<size_t>(X.get_range(i).first, X.get_range(i).second), [&] (const tbb::blocked_range<size_t> r) {
                #pragma omp simd
                for (auto j = r.begin(); j < r.end(); j++) {
                    Y[j] = alpha * X[j] + Y[j];
                }
            }, part);
        });
    }
}

BENCHMARK(benchTransformOmpNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformOmpNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformOmpNestingNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformOmpNestingNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbNoInit)->Apply(Args)->UseRealTime();
BENCHMARK(benchTransformTbbNoInit2)->Apply(Args)->UseRealTime();
BENCHMARK_MAIN();