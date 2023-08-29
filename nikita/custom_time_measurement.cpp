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

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>

#define numberOfRepetitions 50
#define LOWER_LIMIT 15
#define UPPER_LIMIT 33

using ValueType = float;

void writeToFile(std::string dataOutput, std::string filename)
{
	std::fstream file;
    file.open(filename, std::ios_base::app);
 
    if(!file.is_open())
    {
        std::cout << "Unable to open the file" << std::endl;
    }
	else
	{
		dataOutput += "\n";
    	file << dataOutput;
    	file.close();
	}
}

void benchReduceHPXTesting(long dataSize)
{
	std::cout << "Running hypthetical benchmark on locality " << hpx::get_locality_id() << std::endl;
}

int hpx_main(int argc, char* argv[])
{
	// Create a barrier with the desired number of localities
	hpx::barrier b(hpx::find_all_localities().size());

	// Später dann thread number
	if(hpx::get_locality_id() == 0)
	{
		for (int i = LOWER_LIMIT ; i < UPPER_LIMIT; i++)
		{
			// Using time point and system_clock
			std::chrono::time_point<std::chrono::system_clock> start, end;
			std::chrono::duration<double> diff;
			std::vector<std::chrono::duration<double>> durations;

			// Run all the iterations
			long dataSize = std::pow(2, i);
			for (int j = 0 ; j < numberOfRepetitions ; j++)
			{
				start = std::chrono::system_clock::now();

				benchReduceHPXTesting(dataSize);

				end = std::chrono::system_clock::now();

				diff = end - start;
				durations.push_back(diff);

				// Wait for other localities
				b.arrive_and_wait();
			}
			
			// Calculate the mean
			double totalSeconds = 0.0;
			for (const auto& duration : durations)
			{
				totalSeconds += duration.count();
			}
			double meanSeconds = totalSeconds / durations.size();

			// Calculate the standard deviation
			double sumSquaredDifferences = 0.0;
			for (const auto& duration : durations)
			{
				double difference = duration.count() - meanSeconds;
				sumSquaredDifferences += difference * difference;
			}
			double standardDeviation = std::sqrt(sumSquaredDifferences / durations.size());

			// Convert the mean and standard deviation back to durations if needed
			std::chrono::duration<double> meanDuration(meanSeconds);
			std::chrono::duration<double> stdDevDuration(standardDeviation);
		
			std::string output = std::to_string(dataSize) + "," + std::to_string(meanDuration.count()) + "," + std::to_string(stdDevDuration.count());
			writeToFile(output, "test.txt");
		}
	}
	else
	{
		for (int i = LOWER_LIMIT ; i < UPPER_LIMIT; i++)
		{
			// Run all the iterations
			long dataSize = std::pow(2, i);
			for (int j = 0 ; j < numberOfRepetitions ; j++)
			{
				benchReduceHPXTesting(dataSize);

				// Wait for other localities
				b.arrive_and_wait();
			}
		}
	}

	std::cout << "Locality " << hpx::get_locality_id() << " is completly done" << std::endl;
	return hpx::finalize();
}

int main(int argc, char* argv[])
{
	// run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    hpx::init_params init_args;
    init_args.cfg = cfg;

	if (std::filesystem::exists("test.txt")) {
        std::filesystem::remove("test.txt");
        std::cout << "File deleted." << std::endl;
    }

	return hpx::init(argc, argv, init_args);
}

