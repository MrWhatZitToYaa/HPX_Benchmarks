#pragma once

#include <vector>
#include <utility>
#include <omp.h>

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <oneapi/tbb/info.h>


using namespace oneapi;

namespace numa {

class ArenaMgtTBBV3 {
    public:
        ArenaMgtTBBV3() {
            numa_ids = tbb::info::numa_nodes();
            nodes = numa_ids.size();
            task_arenas.resize(nodes);
            task_groups = std::move(std::vector<tbb::task_group>{static_cast<unsigned long>(nodes)});

            omp_set_dynamic(0);
            omp_set_num_threads(nodes);

            #pragma omp parallel proc_bind(spread)
            {
                auto i = omp_get_place_num();
                task_arenas[i].initialize(tbb::task_arena::constraints{numa_ids[i]});
            }
        }

        ArenaMgtTBBV3(int max_concurrency) {
            numa_ids = tbb::info::numa_nodes();
            nodes = numa_ids.size();
            task_arenas.resize(nodes);
            task_groups = std::move(std::vector<tbb::task_group>{static_cast<unsigned long>(nodes)});

            omp_set_dynamic(0);
            omp_set_num_threads(nodes);

            #pragma omp parallel proc_bind(spread)
            {
                auto i = omp_get_place_num();
                task_arenas[i].initialize(tbb::task_arena::constraints{numa_ids[i], max_concurrency});
            }
        }

        int get_nodes() {
            return nodes;
        }

        template <typename F>
        inline void execute(F&& func) {
            #pragma omp parallel proc_bind(spread)
            {
                auto i = omp_get_place_num();
                task_arenas[i].execute([&, i] () {
                    task_groups[i].run_and_wait([&, i] () {
                        func(i);
                    });
                });
            }
        }

        template <typename F>
        inline void enqueue(F&& func) {
            #pragma omp parallel for proc_bind(spread)
            for (size_t i = 0; i < nodes; i++) {
                task_arenas[i].execute([&, i] () {
                    task_groups[i].run([&, i] () {
                        func(i);
                    });
                });
            }
        }

        template <typename F>
        inline void execute_on_node(int node, F&& func) {
            task_arenas[node].execute([&] () {
                task_groups[node].run_and_wait([&] () {
                    func();
                });
            });
        }

        void wait() {
            #pragma omp parallel for proc_bind(spread)
            for (auto i = 0; i < nodes; i++) {
                task_arenas[i].execute([&, i] () {
                    task_groups[i].wait();
                });
            }
            #pragma omp barrier
        }

    private:
        std::vector<tbb::numa_node_id> numa_ids;
        std::vector<tbb::task_group> task_groups;
        std::vector<tbb::task_arena> task_arenas;

        int nodes;
};

}