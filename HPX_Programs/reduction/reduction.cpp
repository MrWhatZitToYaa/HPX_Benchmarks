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

hpx::init_params init_args;
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
    VALUETYPE size = vm["maxelems"].as<VALUETYPE>();
    int loop_count = vm["loop_count"].as<int>();
    int warmup_loop_count = vm["warmup_loop_count"].as<int>();
    
    //print vector size
    if (0 == hpx::get_locality_id())
    {
        hpx::cout << "Reduction Vector Size: " << size << "\n" << std::flush;
    }
 
    char const* const vector_name_1 =
        "v_vector";
    char const* const vector_name_2 =
        "sums_per_locality_vector";
 
    {
        // create vector on one locality, connect to it from all others
        hpx::partitioned_vector<VALUETYPE> v;
        hpx::partitioned_vector<VALUETYPE> sums_per_locality;
        
        if (0 == hpx::get_locality_id())
        {
            std::vector<hpx::id_type> localities = hpx::find_all_localities();
            
            v = hpx::partitioned_vector<VALUETYPE>(
                                              size, hpx::container_layout(localities));
            v.register_as(vector_name_1);

            sums_per_locality = hpx::partitioned_vector<VALUETYPE>(localities.size(), hpx::container_layout(localities));
            sums_per_locality.register_as(vector_name_2);
        }
        else
        {
            hpx::future<void> f1 = v.connect_to(vector_name_1);
            hpx::future<void> f2 = sums_per_locality.connect_to(vector_name_2);
            f1.get();
            f2.get();
        }

        // fill vector v with numbers 2
        partitioned_vector_view<VALUETYPE> view_v(v);
        hpx::generate(hpx::execution::par, view_v.begin(), view_v.end(),
                      [&]() { return 2; });
        
        partitioned_vector_view<VALUETYPE> view_sums_per_locality(sums_per_locality);
        
        for (int round = 1; round <= warmup_loop_count; ++round) {
            VALUETYPE result = hpx::reduce(hpx::execution::par, view_v.begin() , view_v.end());
            view_sums_per_locality[0] = result;
            
            // Wait for all localities to reach this point.
            hpx::distributed::barrier::synchronize();
            if (0 == hpx::get_locality_id())
            {
                VALUETYPE result = hpx::reduce(hpx::execution::par, sums_per_locality.begin() , sums_per_locality.end());
            }
        }
        
        //start timer
        hpx::chrono::high_resolution_timer t;
        
        for (int round = 1; round <= loop_count; ++round) {
            VALUETYPE result = hpx::reduce(hpx::execution::par, view_v.begin() , view_v.end());
            view_sums_per_locality[0] = result;

            //hpx::cout << "locality: " << hpx::get_locality_id() <<  ", Reduction: " << result << "\n" << std::flush;
            
            // Wait for all localities to reach this point.
            hpx::distributed::barrier::synchronize();

            if (0 == hpx::get_locality_id())
            {
                VALUETYPE result = hpx::reduce(hpx::execution::par, sums_per_locality.begin() , sums_per_locality.end());
                //hpx::cout << "result: " << result << "\n" << std::flush;
            }
        }
        
        //end timer
        double elapsed = t.elapsed() / loop_count;
        hpx::util::format_to(std::cout,
                "Elapsed Time == {1} [s]\n",
                elapsed);
    }
         
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("maxelems,m",
         value<VALUETYPE>()->default_value(1024)
         ,"size of the vector")
    
        ("loop_count"
        , hpx::program_options::value<int>()->default_value(10)
        , "number of rounds in performance measurement loop")
    
        ("warmup_loop_count"
        , hpx::program_options::value<int>()->default_value(4)
        , "number of warmup rounds in cache warmup loop")
        ;

    // run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif

