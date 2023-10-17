#pragma once

#include <vector>
#include <utility>
#include <concepts>
#include <ranges>
#include <execution>
#include <omp.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>

#include "allocator_adaptor.hpp"
#include "arenaV3.hpp"

template <typename Container>
concept container = std::random_access_iterator<typename Container::iterator> || std::contiguous_iterator<typename Container::iterator> || 
                    std::ranges::view<Container>;

template <typename T, 
          container Container = std::vector<T, numa::no_init_allocator<T>>>
class numa_adaptor {
public:
    using container_type = Container;
    using value_type = typename Container::value_type;
    using size_type = typename Container::size_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;

    // constructors

    numa_adaptor() : container_(Container()) { }

    explicit numa_adaptor(size_type count) : container_(Container(count)) { }

    explicit numa_adaptor(size_type count, const T& value) : container_(Container(count)) {
        #pragma omp parallel for
        for (size_t i = 0; i < count; i++) {
            new(&container_[i]) value_type{value};
        }        
    }
    
    explicit numa_adaptor(size_type count, const T& value, numa::ArenaMgtTBBV3& arena) : container_(Container(count)) {
        node_ranges(count, arena.get_nodes());
        arena.execute([&] (const int i) {
            tbb::parallel_for(tbb::blocked_range<size_t>(node_range[i].first, node_range[i].second), [&] (const tbb::blocked_range<size_t> r) {
                #pragma omp simd
                for (auto i = r.begin(); i < r.end(); i++) {
                    new(&container_[i]) value_type{value};
                }
            });
        });
    }
    
    explicit numa_adaptor(size_type count, const T& value, int nodes) : container_(Container(count)) {
        node_ranges(count, nodes);
        #pragma omp parallel proc_bind(spread) num_threads(nodes)
        {
            auto i = omp_get_place_num();
            std::uninitialized_fill(std::execution::unseq, container_.begin() + node_range[i].first, container_.begin() + node_range[i].second, value);
        }
    }

    ~numa_adaptor() = default;

    constexpr size_type size() const noexcept { return container_.size(); }

    constexpr T* data() noexcept { return container_.data(); }

    constexpr iterator begin() noexcept { return container_.begin(); }

    constexpr iterator end() noexcept { return container_.end(); }

    constexpr const_iterator cbegin() const noexcept { return container_.cbegin(); }

    constexpr const_iterator cend() const noexcept { return container_.cend(); }

    reference operator[](size_type index) { return container_[index]; }

    reference back() { return container_.back(); };

    std::pair<size_t, size_t> get_range(size_type index) { return node_range[index]; }

private:
    void node_ranges(size_type count, int nodes) {
        node_range.reserve(nodes);

        auto node_size = count / nodes;
        auto rest = count % nodes;

        size_t start = 0;
        for (auto i = 0; i < nodes; i++) {
            auto size = (i < rest) ? node_size + 1 : node_size; 
            node_range[i] = std::make_pair(start, start + size);
            start += size;
        }
    }

    Container container_;

    std::vector<std::pair<size_t, size_t>> node_range;

};