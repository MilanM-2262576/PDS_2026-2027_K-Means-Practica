#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <utility>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"

#include <vector>
#include <limits>
#include <stdexcept>
#include <cfloat>
#include <cuda.h>

void usage()
{
	std::cerr << R"XYZ(
Usage:

  kmeans --input inputfile.csv --output outputfile.csv --k numclusters --repetitions numrepetitions --seed seed [--blocks numblocks] [--threads numthreads] [--trace clusteridxdebug.csv] [--centroidtrace centroiddebug.csv]

Arguments:

 --input:
 
   Specifies input CSV file, number of rows represents number of points, the
   number of columns is the dimension of each point.

 --output:

   Output CSV file, just a single row, with as many entries as the number of
   points in the input file. Each entry is the index of the cluster to which
   the point belongs. The script 'visualize_clusters.py' can show this final
   clustering.

 --k:

   The number of clusters that should be identified.

 --repetitions:

   The number of times the k-means algorithm is repeated; the best clustering
   is kept.

 --blocks:

   Only relevant in CUDA version, specifies the number of blocks that can be
   used.

 --threads:

   Not relevant for the serial version. For the OpenMP version, this number 
   of threads should be used. For the CUDA version, this is the number of 
   threads per block. For the MPI executable, this should be ignored, but
   the wrapper script 'mpiwrapper.sh' can inspect this to run 'mpirun' with
   the correct number of processes.

 --seed:

   Specifies a seed for the random number generator, to be able to get 
   reproducible results.

 --trace:

   Debug option - do NOT use this when timing your program!

   For each repetition, the k-means algorithm goes through a sequence of 
   increasingly better cluster assignments. If this option is specified, this
   sequence of cluster assignments should be written to a CSV file, similar
   to the '--output' option. Instead of only having one line, there will be
   as many lines as steps in this sequence. If multiple repetitions are
   specified, only the results of the first repetition should be logged
   for clarity. The 'visualize_clusters.py' program can help to visualize
   the data logged in this file.

 --centroidtrace:

   Debug option - do NOT use this when timing your program!

   Should also only log data during the first repetition. The resulting CSV 
   file first logs the randomly chosen centroids from the input data, and for
   each step in the sequence, the updated centroids are logged. The program 
   'visualize_centroids.py' can be used to visualize how the centroids change.
   
)XYZ";
	exit(-1);
}

void readData(std::ifstream &input, std::vector<double> &allData, size_t &numRows, size_t &numCols)
{
	if (!input.is_open())
		throw std::runtime_error("Input file is not open");

	allData.resize(0);
	numRows = 0;
	numCols = 0;

	CSVReader inReader(input);
	int numColsExpected = -1;
	int line = 1;
	std::vector<double> row;

	while (inReader.read(row))
	{
		if (numColsExpected == -1)
		{
			numColsExpected = row.size();
			if (numColsExpected <= 0)
				throw std::runtime_error("Unexpected error: 0 columns");
		}
		else if (numColsExpected != (int)row.size())
			throw std::runtime_error("Incompatible number of colums read in line " + std::to_string(line) + ": expecting " + std::to_string(numColsExpected) + " but got " + std::to_string(row.size()));

		for (auto x : row)
			allData.push_back(x);

		line++;
	}

	numRows = (size_t)allData.size()/numColsExpected;
	numCols = (size_t)numColsExpected;
}

FileCSVWriter openDebugFile(const std::string &n)
{
	FileCSVWriter f;

	if (n.length() != 0)
	{
		f.open(n);
		if (!f.is_open())
			std::cerr << "WARNING: Unable to open debug file " << n << std::endl;
	}
	return f;
}

__global__ void assign_and_accumulate_kernel(const double *allData, const double *centroids, int *clusters, double *blockSums, int *blockCounts, double *blockDist, int *changed, size_t numRows, size_t numCols, int numClusters)
{
	//geheugen per block
    extern __shared__ double smem[]; // blok geheugen dat we zelf in drie kappen
    double *s_sums = smem; // lijst van double waarden om de som van de punten per cluster bij te houden, om later nieuwe centroids te berekenen(teller)
	double *s_dist = &smem[(size_t)numClusters * numCols]; // een double waarde om de som van de afstanden tot de dichtstbijzijnde cluster bij te houden per block, om te kijken welke repetitie de beste is
 	int *s_counts = (int*)&s_dist[1]; // een int array om het aantal punten per cluster bij te houden,om later nieuwe centroids te berekenen(Noemer)

    // Shared memory op 0 zetten per block
    for (size_t i = threadIdx.x; i < (size_t)numClusters * numCols; i += blockDim.x)
        s_sums[i] = 0.0;
    for (int j = threadIdx.x; j < numClusters; j += blockDim.x)
        s_counts[j] = 0;
    if (threadIdx.x == 0)
        s_dist[0] = 0.0;
    __syncthreads();

	//stride is totaal aantal threads op hele gpu, elke thread pakt zijn eigen punten, en als die klaar is pakt hij de volgende punten die nog niet gedaan zijn
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x; p < numRows; p += stride) {
        int closestCluster = -1;
        double minDistanceSq = DBL_MAX;

		//uit functie find_closest_cenroid_index_and_distace
        for (int j = 0; j < numClusters; ++j) {
            double currentDistSq = 0.0;
            for (size_t d = 0; d < numCols; ++d) {
                double diff = allData[p * numCols + d] - centroids[(size_t)j * numCols + d];
                currentDistSq += diff * diff;
            }

            if (currentDistSq < minDistanceSq) {
                minDistanceSq = currentDistSq;
                closestCluster = j;
            }
        }

		//als cluster veranderd is zet dan changed op 1 zodat de host weet dat er nog een stap nodig is
        if (clusters[p] != closestCluster) {
            clusters[p] = closestCluster;
            *changed = 1;
        }

        // voor later nieuwe centroids berekenen, subtotaal per block in shared memory,
        atomicAdd(&s_dist[0], minDistanceSq);
        atomicAdd(&s_counts[closestCluster], 1);
        for (size_t d = 0; d < numCols; ++d) {
            atomicAdd(&s_sums[(size_t)closestCluster * numCols + d], allData[p * numCols + d]);
        }
    }
    __syncthreads();

    // Elk block schrijft zijn eigen subtotaal weg 
    size_t sumOffset = (size_t)blockIdx.x * numClusters * numCols;
    for (size_t i = threadIdx.x; i < (size_t)numClusters * numCols; i += blockDim.x)
        blockSums[sumOffset + i] = s_sums[i];

    size_t countOffset = (size_t)blockIdx.x * numClusters;
    for (int j = threadIdx.x; j < numClusters; j += blockDim.x)
        blockCounts[countOffset + j] = s_counts[j];

    if (threadIdx.x == 0)
        blockDist[blockIdx.x] = s_dist[0];
}

int kmeans(Rng &rng, const std::string &inputFile, const std::string &outputFileName, int numClusters, int repetitions, int numBlocks, int numThreads, const std::string &centroidDebugFileName, const std::string &clusterDebugFileName)
{
	// If debug filenames are specified, this opens them. The is_open method
	// can be used to check if they are actually open and should be written to.
	FileCSVWriter centroidDebugFile = openDebugFile(centroidDebugFileName);
	FileCSVWriter clustersDebugFile = openDebugFile(clusterDebugFileName);

	FileCSVWriter csvOutputFile(outputFileName);
	if (!csvOutputFile.is_open())
	{
		std::cerr << "Unable to open output file " << outputFileName << std::endl;
		return -1;
	}

	// TODO: load dataset
	std::ifstream inFile(inputFile);
    if (!inFile.is_open())
    {
        std::cerr << "Unable to open input file " << inputFile << std::endl;
        return -1;
    }

    std::vector<double> allData;
    size_t numRows = 0, numCols = 0;
    readData(inFile, allData, numRows, numCols);
    inFile.close();

	// GPU al eens eerste keer initialiseren, zodat de eerste cudaMalloc niet in de tijdmeting zit
	cudaFree(0);

	Timer timer;

	// GPU-geheugen alloceren (nu per block zodat de optelvolgorde op CPU vast ligt)
	double *d_data, *d_centroids, *d_blockSums, *d_blockDist;
	int *d_clusters, *d_blockCounts, *d_changed;

	size_t blockSumsSize = (size_t)numBlocks * numClusters * numCols;
	size_t blockCountsSize = (size_t)numBlocks * numClusters;

	cudaMalloc(&d_data, numRows * numCols * sizeof(double));
	cudaMalloc(&d_centroids, numClusters * numCols * sizeof(double));
	cudaMalloc(&d_blockSums, blockSumsSize * sizeof(double));
	cudaMalloc(&d_clusters, numRows * sizeof(int));
	cudaMalloc(&d_blockCounts, blockCountsSize * sizeof(int));
	cudaMalloc(&d_blockDist, numBlocks * sizeof(double));
	cudaMalloc(&d_changed, sizeof(int));

	// Dataset eenmalig naar de GPU kopieren
	cudaMemcpy(d_data, allData.data(), allData.size() * sizeof(double), cudaMemcpyHostToDevice);

	// Host-buffers
	std::vector<double> h_centroids(numClusters * numCols);
	std::vector<double> h_blockSums(blockSumsSize);
	std::vector<int> h_blockCounts(blockCountsSize);
	std::vector<double> h_blockDist(numBlocks);
	std::vector<int> h_clusters(numRows);

	// Hoeveel shared memory 1 block nodig heeft
	size_t sharedMemBytes = ((size_t)numClusters * numCols + 1) * sizeof(double) + (size_t)numClusters * sizeof(int);

	std::vector<int> bestClusters;
	double bestDistSquaredSum = std::numeric_limits<double>::max();
	std::vector<size_t> stepsPerRepetition(repetitions);

	for (int r = 0 ; r < repetitions ; r++)
	{
		size_t numSteps = 0;

		std::vector<size_t> indices(numClusters);
		rng.pickRandomIndices(numRows, indices);

		for (int j = 0; j < numClusters; ++j)
			for (size_t d = 0; d < numCols; ++d)
				h_centroids[j * numCols + d] = allData[indices[j] * numCols + d];

		cudaMemset(d_clusters, -1, numRows * sizeof(int));

		double distanceSquaredSum = 0.0;
		int h_changed = 1;

		while (h_changed) {
			// centroids en changed naar GPU kopieren
			cudaMemset(d_changed, 0, sizeof(int));
			cudaMemcpy(d_centroids, h_centroids.data(), h_centroids.size() * sizeof(double), cudaMemcpyHostToDevice);

			// Kernel uitvoeren met shared memory
			assign_and_accumulate_kernel<<<numBlocks, numThreads, sharedMemBytes>>>(
				d_data, d_centroids, d_clusters, d_blockSums, d_blockCounts, d_blockDist, d_changed,
				numRows, numCols, numClusters
			);

			// Resultaten van de kernel terug naar de host kopieren
			cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_blockDist.data(), d_blockDist, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_blockSums.data(), d_blockSums, blockSumsSize * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_blockCounts.data(), d_blockCounts, blockCountsSize * sizeof(int), cudaMemcpyDeviceToHost);


			distanceSquaredSum = 0.0; //ook berekenen om te kijken of rep beter is
			std::vector<double> totalSums(numClusters * numCols, 0.0);
			std::vector<int> totalCounts(numClusters, 0);

			// subtotaal van alle blocks optellen, zodat we de nieuwe centroids kunnen berekenen
			for (int b = 0; b < numBlocks; ++b) {
				distanceSquaredSum += h_blockDist[b];
				for (int j = 0; j < numClusters; ++j) {
					totalCounts[j] += h_blockCounts[b * numClusters + j];
					for (size_t d = 0; d < numCols; ++d) {
						totalSums[j * numCols + d] += h_blockSums[(b * numClusters + j) * numCols + d];
					}
				}
			}

			if (clustersDebugFile.is_open()) {
				cudaMemcpy(h_clusters.data(), d_clusters, h_clusters.size() * sizeof(int), cudaMemcpyDeviceToHost);
				clustersDebugFile.write(h_clusters);
			}

			if (centroidDebugFile.is_open()) {
				for (int j = 0; j < numClusters; j++) {
					std::vector<double> c_row(h_centroids.begin() + j * numCols, h_centroids.begin() + (j + 1) * numCols);
					centroidDebugFile.write(c_row);
				}
			}

			// als er een verandering is geweest(een punt heeft een andere cluster toegewezen gekregen), bereken de nieuwe centroids
			if (h_changed) {
				for (int j = 0; j < numClusters; j++) {
					if (totalCounts[j] > 0) {
						for (size_t d = 0; d < numCols; d++)
							h_centroids[j * numCols + d] = totalSums[j * numCols + d] / static_cast<double>(totalCounts[j]);
					}
				}
			}

			numSteps++;
		}

		if (distanceSquaredSum < bestDistSquaredSum) {
			bestDistSquaredSum = distanceSquaredSum;
			cudaMemcpy(h_clusters.data(), d_clusters, h_clusters.size() * sizeof(int), cudaMemcpyDeviceToHost);
			bestClusters = h_clusters;
		}

		stepsPerRepetition[r] = numSteps;

		centroidDebugFile.close();
		clustersDebugFile.close();
	}

	timer.stop();

	cudaFree(d_data);
	cudaFree(d_centroids);
	cudaFree(d_blockSums);
	cudaFree(d_clusters);
	cudaFree(d_blockCounts);
	cudaFree(d_blockDist);
	cudaFree(d_changed);

	std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
	std::cout << "cuda," << numBlocks << "," << numThreads << "," << inputFile << "," 
			  << rng.getUsedSeed() << "," << numClusters << ","
		      << repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds()/1e9
			  << std::endl;

	csvOutputFile.write(stepsPerRepetition, "# Steps: ");
	csvOutputFile.write( bestClusters );
	return 0;
}

int mainCxx(const std::vector<std::string> &args)
{
	if (args.size()%2 != 0)
		usage();

	std::string inputFileName, outputFileName, centroidTraceFileName, clusterTraceFileName;
	unsigned long seed = 0;

	int numClusters = -1, repetitions = -1;
	int numBlocks = 1, numThreads = 1;
	for (int i = 0 ; i < (int)args.size() ; i += 2)
	{
		if (args[i] == "--input")
			inputFileName = args[i+1];
		else if (args[i] == "--output")
			outputFileName = args[i+1];
		else if (args[i] == "--centroidtrace")
			centroidTraceFileName = args[i+1];
		else if (args[i] == "--trace")
			clusterTraceFileName = args[i+1];
		else if (args[i] == "--k")
			numClusters = stoi(args[i+1]);
		else if (args[i] == "--repetitions")
			repetitions = stoi(args[i+1]);
		else if (args[i] == "--seed")
			seed = stoul(args[i+1]);
		else if (args[i] == "--blocks")
			numBlocks = stoi(args[i+1]);
		else if (args[i] == "--threads")
			numThreads = stoi(args[i+1]);
		else
		{
			std::cerr << "Unknown argument '" << args[i] << "'" << std::endl;
			return -1;
		}
	}

	if (inputFileName.length() == 0 || outputFileName.length() == 0 || numClusters < 1 || repetitions < 1 || seed == 0)
		usage();

	Rng rng(seed);

	return kmeans(rng, inputFileName, outputFileName, numClusters, repetitions,
			      numBlocks, numThreads, centroidTraceFileName, clusterTraceFileName);
}

int main(int argc, char *argv[])
{
	std::vector<std::string> args;
	for (int i = 1 ; i < argc ; i++)
		args.push_back(argv[i]);

	return mainCxx(args);
}