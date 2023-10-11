#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <hpx/iostream.hpp>

///////////////////////////////////////////////////////////////////////////////
using VALUETYPE = float;

// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(VALUETYPE)

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
int hpx_main(hpx::program_options::variables_map& vm)
{
    VALUETYPE size = 10;
    if (vm.count("maxelems"))
        size = vm["maxelems"].as<VALUETYPE>();
    
    hpx::cout << "Transform Vector Size: " << size << "\n" << std::flush;

    char const* const vector_name_v = "v_vector";
    char const* const vector_name_y = "y_vector";
    char const* const latch_name = "latch_spmd";

    {
        // create vector on one locality, connect to it from all others
        hpx::partitioned_vector<VALUETYPE> v;
        hpx::partitioned_vector<VALUETYPE> y;
        hpx::distributed::latch l;

        if (0 == hpx::get_locality_id())
        {
            std::vector<hpx::id_type> localities = hpx::find_all_localities();

            v = hpx::partitioned_vector<VALUETYPE>(
                size, hpx::container_layout(localities));
            v.register_as(vector_name_v);

            y = hpx::partitioned_vector<VALUETYPE>(
                size, hpx::container_layout(localities));
            y.register_as(vector_name_y);

            l = hpx::distributed::latch(localities.size());
            l.register_as(latch_name);
        }
        else
        {
            hpx::future<void> f1 = v.connect_to(vector_name_v);
            hpx::future<void> f2 = y.connect_to(vector_name_y);
            l.connect_to(latch_name);
            f1.get();
            f2.get();
        }

        // fill the vector v with 1
        partitioned_vector_view<VALUETYPE> view_v(v);
        hpx::generate(hpx::execution::par, view_v.begin(), view_v.end(),
            [&]() { return 1; });
        
        /*
        for(int i = 0; i<v.size(); i++){
            hpx::cout << "RANDOM v: " << v[i] << "\n" << std::flush;
        }
        */

        // fill the vector y with 2
        partitioned_vector_view<VALUETYPE> view_y(y);
        hpx::generate(hpx::execution::par, view_y.begin(), view_y.end(),
            [&]() { return 2; });
        
        /*
        for(int i = 0; i<y.size(); i++){
            hpx::cout << "RANDOM y: " << y[i] << "\n" << std::flush;
        }
        */
        
        // Transform the values of view_v by adding the corresponding values from view_y
        hpx::transform(hpx::execution::par, view_v.begin(), view_v.end(), view_y.begin(), view_v.begin(),
                   [](VALUETYPE v, VALUETYPE y) { return v + y; });
        
        /*
        for(int i = 0; i<view_v.size(); i++){
            hpx::cout << "locality: " << hpx::get_locality_id() <<  ", Addition: " << view_v[i] << "\n" << std::flush;
        }

        */

       
        /*
        // do the same using a plain loop
        std::size_t maxnum = view_v.size();
        for (std::size_t i = 0; i != maxnum; ++i)
            view_v[i] += view_y[i];

        for(int i = 0; i<v.size(); i++){
            hpx::cout << "*2 WITH LOOP: " << v[i] << "\n" << std::flush;
        }
        */

        // Wait for all localities to reach this point.
        l.arrive_and_wait();
    }
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
        ("maxelems,m", value<VALUETYPE>(),
            "the data array size to use (default: 10000)")
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
#endif
