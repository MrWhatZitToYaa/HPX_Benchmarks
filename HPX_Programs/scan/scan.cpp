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
        std::cout << "Scan Vector Size: " << size << std::endl;
    }
    
    char const* const vector_name_1 =
        "partitioned_vector_1";
    char const* const vector_name_2 =
        "partitioned_vector_2";
    
    {
        // create vector on one locality, connect to it from all others
        hpx::partitioned_vector<VALUETYPE> main_vector;
        hpx::partitioned_vector<VALUETYPE> sums_per_locality;
        
        if (0 == hpx::get_locality_id())
        {
            std::vector<hpx::id_type> localities = hpx::find_all_localities();
            
            main_vector = hpx::partitioned_vector<VALUETYPE>(size, hpx::container_layout(localities));
            main_vector.register_as(vector_name_1);
 
            sums_per_locality = hpx::partitioned_vector<VALUETYPE>(localities.size(), hpx::container_layout(localities));
            sums_per_locality.register_as(vector_name_2);
        }
        else
        {
            hpx::future<void> f1 = main_vector.connect_to(vector_name_1);
            hpx::future<void> f2 = sums_per_locality.connect_to(vector_name_2);
            f1.get();
            f2.get();
        }
 
        // fill the partitioned vector main_vector with numbers 2 per locality
        partitioned_vector_view<VALUETYPE> main_vector_view(main_vector);
        hpx::generate(hpx::execution::par, main_vector_view.begin(), main_vector_view.end(),
                      [&]() { return 2; });
        
        // Situation example (main_vector):
        // 3 Localities (Lx) and a vector size of 15:
        // L0 main_vector_view(5) 2 2 2 2 2
        // L1 main_vector_view(5)           2 2 2 2 2
        // L2 main_vector_view(5)                     2 2 2 2 2
        // main_vector:           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        
        partitioned_vector_view<VALUETYPE> sums_per_locality_view(sums_per_locality);
        
        // warm-up cache
        for (int round = 1; round <= warmup_loop_count; ++round) {
            // reduce per locality the main_vector entries and save it via sums_per_locality_view:
            VALUETYPE result = hpx::reduce(hpx::execution::par, main_vector_view.begin() , main_vector_view.end());
            sums_per_locality_view[0] = result;
            
            // Situation example (sums_per_locality):
            // 3 Localities (Lx):
            // L0 sums_per_locality_view[0]        10
            // L1 sums_per_locality_view[0]           10
            // L2 sums_per_locality_view[0]              10
            // sums_per_locality:                  10 10 10
            
            // Wait for all localities to reach this point.
            hpx::distributed::barrier::synchronize();
        
            if (0 == hpx::get_locality_id())
            {
                //sums_per_locality: 10 10 10 --> has to be changed to 10 20 30 via inclusive_scan (locality 0 has access to the whole vector):
                hpx::inclusive_scan(hpx::execution::par, sums_per_locality.begin(), sums_per_locality.end(), sums_per_locality.begin());
                
                //now we have to shift_right the transformed sums_per_locality --> from 10 20 30 to 10 10 20:
                for (VALUETYPE i = sums_per_locality.size()-1; i != 0; --i)
                {
                     VALUETYPE x = sums_per_locality[i-1];
                     sums_per_locality[i] = x;
                }
                // finally change it from 10 10 20 to 0 10 20:
                sums_per_locality[0] = 0;
            }
            
            // Wait for all localities to reach this point.
            hpx::distributed::barrier::synchronize();
            
            // make the final inclusive_scan on the main_vector_view and start with the respective value from sums_per_locality via the sums_per_locality_view:
            hpx::inclusive_scan(hpx::execution::par, main_vector_view.begin(), main_vector_view.end(), main_vector_view.begin(), std::plus<VALUETYPE>(), sums_per_locality_view[0]);
            
            }
        
        //start timer
        hpx::chrono::high_resolution_timer t;
        for (int round = 1; round <= loop_count; ++round) {
            VALUETYPE result = hpx::reduce(hpx::execution::par, main_vector_view.begin() , main_vector_view.end());
            sums_per_locality_view[0] = result;
            
            // Wait for all localities to reach this point.
            hpx::distributed::barrier::synchronize();
        
            if (0 == hpx::get_locality_id())
            {
                hpx::inclusive_scan(hpx::execution::par, sums_per_locality.begin(), sums_per_locality.end(), sums_per_locality.begin());
                
                for (VALUETYPE i = sums_per_locality.size()-1; i != 0; --i)
                {
                     VALUETYPE x = sums_per_locality[i-1];
                     sums_per_locality[i] = x;
                }
                
                sums_per_locality[0] = 0;
            }
            
            // Wait for all localities to reach this point.
            hpx::distributed::barrier::synchronize();
            
            hpx::inclusive_scan(hpx::execution::par, main_vector_view.begin(), main_vector_view.end(), main_vector_view.begin(), std::plus<VALUETYPE>(), sums_per_locality_view[0]);
            
            }
        //end timer
        double elapsed = t.elapsed() / loop_count;
        hpx::util::format_to(std::cout,
                "Elapsed Time == {1} [s]\n",
                elapsed);
 
        // Wait for all localities to reach this point.
        hpx::distributed::barrier::synchronize();
         
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
