#include <hpx/local/algorithm.hpp>
#include <hpx/local/future.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/memory.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/config.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>

#include <benchmark/benchmark.h>

using ValueType = float;

static void Args(benchmark::internal::Benchmark* b) {
  const int64_t lowerLimit = 3;
  const int64_t upperLimit = 10;

  for (auto x = lowerLimit; x <= upperLimit; ++x) {
        b->Args({int64_t{1} << x});
  }
}

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

void benchReduceHPX(/*benchmark::State& state*/)
{
    unsigned int size = 5;//state.range(0);

    char const* const example_vector_name = "partitioned_vector_spmd_reduction";
    char const* const example_latch_name  = "latch_spmd_reduction";

    {
        // create vector on one locality, connect to it from all others
        hpx::partitioned_vector<ValueType> v;
        hpx::distributed::latch l;

        if (0 == hpx::get_locality_id())
        {
            std::vector<hpx::id_type> localities = hpx::find_all_localities();

			auto layout = hpx::container_layout(localities);
            v = hpx::partitioned_vector<ValueType>(size, layout);
            v.register_as(example_vector_name);

            l = hpx::distributed::latch(localities.size());
            l.register_as(example_latch_name);
        }
        else
        {
            hpx::future<void> f1 = v.connect_to(example_vector_name);
            l.connect_to(example_latch_name);
            f1.get();
        }

        // fill the vector with ones
        partitioned_vector_view<ValueType> view(v);
        hpx::generate(hpx::execution::par, view.begin(), view.end(),
            [&]() { return ValueType{1}; });

        // apply reduction operation
		ValueType sum = 1;
		ValueType val1 = 1;

		hpx::experimental::reduction(sum, val1, [](ValueType a, ValueType b) { return a+b; });
		std::cout << val1 << std::endl;
        // Wait for all localities to reach this point.
        l.arrive_and_wait();
    }

	/*
	for(auto _ : state)
	{
		sum = 0;
	}
	*/
}

int main(int argc, char* argv[])
{
	//::benchmark::Initialize(&argc, argv);
	//::benchmark::RunSpecifiedBenchmarks();
	benchReduceHPX();
    return 0;
}

//BENCHMARK(benchReduceHPX)->Apply(Args)->UseRealTime();