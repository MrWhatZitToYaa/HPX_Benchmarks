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
    VALUETYPE size = 15;
    if (vm.count("maxelems"))
        size = vm["maxelems"].as<VALUETYPE>();
    
    //print vector size
    if (0 == hpx::get_locality_id())
    {
        std::cout << "using vector size: " << size << std::endl;
    }
    
    char const* const vector_name_1 =
        "partitioned_vector_1";
    char const* const vector_name_2 =
        "partitioned_vector_2";
    char const* const latch_reduction_name = "latch_reduction_name";
    char const* const latch_first_scan_name = "latch_first_scan_name";
    char const* const latch_shifted_sums_name = "latch_shifted_sums_name";
    char const* const latch_transform_main_vector_name = "latch_transform_main_vector_name";
    
    {
        // create vector on one locality, connect to it from all others
        hpx::partitioned_vector<VALUETYPE> main_vector;
        hpx::partitioned_vector<VALUETYPE> sums_per_locality;
        hpx::distributed::latch latch_reduction;
        hpx::distributed::latch latch_first_scan;
        hpx::distributed::latch latch_shifted_sums;
        hpx::distributed::latch latch_transform_main_vector;
        
        if (0 == hpx::get_locality_id())
        {
            std::vector<hpx::id_type> localities = hpx::find_all_localities();
            
            main_vector = hpx::partitioned_vector<VALUETYPE>(size, hpx::container_layout(localities));
            main_vector.register_as(vector_name_1);
 
            sums_per_locality = hpx::partitioned_vector<VALUETYPE>(localities.size(), hpx::container_layout(localities));
            sums_per_locality.register_as(vector_name_2);
            
            latch_reduction = hpx::distributed::latch(localities.size());
            latch_reduction.register_as(latch_reduction_name);
            
            latch_first_scan = hpx::distributed::latch(localities.size());
            latch_first_scan.register_as(latch_first_scan_name);
            
            latch_shifted_sums = hpx::distributed::latch(localities.size());
            latch_shifted_sums.register_as(latch_shifted_sums_name);
            
            latch_transform_main_vector = hpx::distributed::latch(localities.size());
            latch_transform_main_vector.register_as(latch_transform_main_vector_name);
        }
        else
        {
            hpx::future<void> f1 = main_vector.connect_to(vector_name_1);
            hpx::future<void> f2 = sums_per_locality.connect_to(vector_name_2);
            latch_reduction.connect_to(latch_reduction_name);
            latch_first_scan.connect_to(latch_first_scan_name);
            latch_shifted_sums.connect_to(latch_shifted_sums_name);
            latch_transform_main_vector.connect_to(latch_transform_main_vector_name);
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
        
        // reduce per locality the main_vector entries and save it via sums_per_locality_view:
        partitioned_vector_view<VALUETYPE> sums_per_locality_view(sums_per_locality);
        VALUETYPE result = hpx::reduce(hpx::execution::par, main_vector_view.begin() , main_vector_view.end());
        sums_per_locality_view[0] = result;
        
        // Situation example (sums_per_locality):
        // 3 Localities (Lx):
        // L0 sums_per_locality_view[0]        10
        // L1 sums_per_locality_view[0]           10
        // L2 sums_per_locality_view[0]              10
        // sums_per_locality:                  10 10 10
        
        // Wait for all localities to reach this point.
        latch_reduction.arrive_and_wait();
 
        // make an inclusive_scan on the main_vector_view:
        hpx::inclusive_scan(hpx::execution::par, main_vector_view.begin(), main_vector_view.end(), main_vector_view.begin());
        
        // Wait for all localities to reach this point.
        latch_first_scan.arrive_and_wait();
        
        // Situation example after inclusive_scan (main_vector):
        // 3 Localities (Lx):
        // L0 main_vector_view(5) 2 4 6 8 10
        // L1 main_vector_view(5)            2 4 6 8 10
        // L2 main_vector_view(5)                       2 4 6 8 10
        // main_vector:           2 4 6 8 10 2 4 6 8 10 2 4 6 8 10

	
        if (0 == hpx::get_locality_id())
        {
            //sums_per_locality: 10 10 10 --> has to be changed to 10 20 30 via inclusive_scan:
            hpx::inclusive_scan(hpx::execution::par, sums_per_locality.begin(), sums_per_locality.end(), sums_per_locality.begin());
            
            //now we have to shift_right the transformed sums_per_locality --> from 10 20 30 to 10 10 20 (first 10 is indifferent):
            for (VALUETYPE i = sums_per_locality.size()-1; i != 0; --i)
            {
                 VALUETYPE x = sums_per_locality[i-1];
                 sums_per_locality[i] = x;
            }
        }
        
        // Wait for all localities to reach this point.
        latch_shifted_sums.arrive_and_wait();
        
        if (0 != hpx::get_locality_id())
        {
            // function to add the specific value of sums_per_locality_view to the main_vector_view value --> for example: Locality 1 values in main_vector_view and on every value, we add a 20, that a "2" is changed to "22"
            auto add_single_value = [&](VALUETYPE input_value) {return input_value + sums_per_locality_view[0];};
            hpx::transform(hpx::execution::par, main_vector_view.begin(), main_vector_view.end(), main_vector_view.begin(), add_single_value);
            
        }
 
        // Wait for all localities to reach this point.
        latch_transform_main_vector.arrive_and_wait();
         
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
            "the data array size to use (default: 15)")
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
