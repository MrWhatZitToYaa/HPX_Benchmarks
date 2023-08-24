#include <hpx/local/algorithm.hpp>
#include <hpx/local/future.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/memory.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/config.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/barrier.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using ValueType = float;

static void Args(benchmark::internal::Benchmark* b) {
  const int64_t lowerLimit = 15;
  const int64_t upperLimit = 33;

  for (auto x = lowerLimit; x <= upperLimit; ++x) {
        b->Args({int64_t{1} << x});
  }
}

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(ValueType)

///////////////////////////////////////////////////////////////////////////////
//
// Define a view for a partitioned vector which exposes the part of the vector
// which is located on the current locality.
//
// This view does not own the data and relies on the partitioned_vector to be
// available during the full lifetime of the view.
//
template <typename T>
struct partitioned_vector_view
{
private:
    typedef typename hpx::partitioned_vector<T>::iterator global_iterator;
    typedef typename hpx::partitioned_vector<T>::const_iterator
        const_global_iterator;

    typedef hpx::traits::segmented_iterator_traits<global_iterator> traits;
    typedef hpx::traits::segmented_iterator_traits<const_global_iterator>
        const_traits;

    typedef typename traits::local_segment_iterator local_segment_iterator;

public:
    typedef typename traits::local_raw_iterator iterator;
    typedef typename const_traits::local_raw_iterator const_iterator;
    typedef T value_type;

public:
    explicit partitioned_vector_view(hpx::partitioned_vector<T>& data)
      : segment_iterator_(data.segment_begin(hpx::get_locality_id()))
    {
#if defined(HPX_DEBUG)
        // this view assumes that there is exactly one segment per locality
        typedef typename traits::local_segment_iterator local_segment_iterator;
        local_segment_iterator sit = segment_iterator_;
        HPX_ASSERT(
            ++sit == data.segment_end(hpx::get_locality_id()));    // NOLINT
#endif
    }

    iterator begin()
    {
        return traits::begin(segment_iterator_);
    }
    iterator end()
    {
        return traits::end(segment_iterator_);
    }

    const_iterator begin() const
    {
        return const_traits::begin(segment_iterator_);
    }
    const_iterator end() const
    {
        return const_traits::end(segment_iterator_);
    }
    const_iterator cbegin() const
    {
        return begin();
    }
    const_iterator cend() const
    {
        return end();
    }

    value_type& operator[](std::size_t index)
    {
        return (*segment_iterator_)[index];
    }
    value_type const& operator[](std::size_t index) const
    {
        return (*segment_iterator_)[index];
    }

    std::size_t size() const
    {
        return (*segment_iterator_).size();
    }

private:
    local_segment_iterator segment_iterator_;
};

///////////////////////////////////////////////////////////////////////////////

void benchReduceHPX(benchmark::State& state)
{
	unsigned int size = state.range(0);
 
    char const* const vector_name_1 =
        "partitioned_vector_1";
    char const* const vector_name_2 =
        "partitioned_vector_2";
    char const* const latch_name = "latch_spmd_foreach";
 
    {
        // create vector on one locality, connect to it from all others
        hpx::partitioned_vector<int> v1;
        hpx::partitioned_vector<int> v2;
        hpx::distributed::latch l;
        
        if (0 == hpx::get_locality_id())
        {
            std::vector<hpx::id_type> localities = hpx::find_all_localities();
            
            v1 = hpx::partitioned_vector<int>(
                                              size, hpx::container_layout(localities));
            v1.register_as(vector_name_1);
            
            v2 = hpx::partitioned_vector<int>(
                                              size, hpx::container_layout(localities));
            v2.register_as(vector_name_2);
            
            l = hpx::distributed::latch(localities.size());
            l.register_as(latch_name);
        }
        else
        {
            hpx::future<void> f1 = v1.connect_to(vector_name_1);
            hpx::future<void> f2 = v2.connect_to(vector_name_2);
            l.connect_to(latch_name);
            f1.get();
            f2.get();
        }
        
        // fill the vector 1 with numbers 1
        partitioned_vector_view<int> view1(v1);
        hpx::generate(hpx::execution::par, view1.begin(), view1.end(),
                      [&]() { return 1; });
        
        // fill the vector 2 with numbers 2
        partitioned_vector_view<int> view2(v2);
        hpx::generate(hpx::execution::par, view2.begin(), view2.end(),
                      [&]() { return 2; });
        
        constexpr int alpha = 2;

		for(auto _ : state)
		{
			// Transform the values of view1 by adding the corresponding values from view2
			benchmark::DoNotOptimize(hpx::transform(hpx::execution::par, view1.begin(), view1.end(), view2.begin(), view1.begin(),
					[](int v, int y) { return alpha * v + y; }));
			/*
			//print the transformed vector partitions from the specific locality
			for(int i = 0; i<view1.size(); i++){
				hpx::cout << "locality: " << hpx::get_locality_id() <<  ", Addition: " << view1[i] << "\n" << std::flush;
			}
			*/
        	l.arrive_and_wait();
		}
        
		/*
        //print the transformed vector partitions from the specific locality
        for(int i = 0; i<view1.size(); i++){
            hpx::cout << "locality: " << hpx::get_locality_id() <<  ", Addition: " << view1[i] << "\n" << std::flush;
        }
        
        //print the transformed vector1
        for(int i = 0; i<v1.size(); i++){
            hpx::cout << "v1: " << v1[i] << "\n" << std::flush;
        }
		*/
    }
}

void benchReduceHPXTesting(benchmark::State& state)
{
	unsigned int size = state.range(0);

	std::string output = "Locality " + std::to_string(hpx::get_locality_id()) + " , before barrier " + "\n";
	std::cout << output;
	hpx::distributed::barrier("Test", 2);
	output = "Locality " + std::to_string(hpx::get_locality_id()) + " , after barrier " + "\n";
	std::cout << output;

	for(auto _ : state)
	{
		output = "Locality " + std::to_string(hpx::get_locality_id()) + " , size " + std::to_string(size) + "\n";
		std::cout << output;
	}
}

int hpx_main(int argc, char* argv[])
{
	/*
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
	*/

	std::string output = "Locality " + std::to_string(hpx::get_locality_id()) + " , before barrier " + "\n";
	std::cout << output;
	//hpx::barrier();
	output = "Locality " + std::to_string(hpx::get_locality_id()) + " , after barrier " + "\n";
	std::cout << output;

	std::cout << "Locality " << hpx::get_locality_id() << " is completly done" << std::endl;
    return hpx::finalize();
}


int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
		("benchmark_time_unit", value<std::string>(),
        "benchmark_time_unit")
		("benchmark_out_format", value<std::string>(),
        "benchmark_out_format")
		("benchmark_out", value<std::string>(),
        "benchmark_out")
		("benchmark_report_aggregates_only", value<std::string>(),
        "benchmark_report_aggregates_only")
		("benchmark_repetitions", value<std::string>(),
        "benchmark_repetitions")
		("benchmark_min_time", value<std::string>(),
        "benchmark_min_time")
		("benchmark_min_warmup_time", value<std::string>(),
        "benchmark_min_warmup_time")
        ;
    // clang-format on

    // run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}

BENCHMARK(benchReduceHPXTesting)->Apply(Args)->UseRealTime();
