#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <utility>
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "rng.h"
#include "timer.h"


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

// Helper function to read input file into allData, setting number of detected
// rows and columns. Feel free to use, adapt or ignore
void readData(std::ifstream &input, std::vector<double> &allData, size_t &numRows, size_t &numCols)
{
	if (!input.is_open())
		throw std::runtime_error("Input file is not open");

	allData.resize(0);
	numRows = 0;
	numCols = -1;

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

std::pair<int, double> find_closest_centroid_index_and_distance(size_t rowNumber, const std::vector<double> &allData, size_t numCols, const std::vector<std::vector<double>> &centroids)
{
    int closestCluster = -1;
    double minDistanceSq = std::numeric_limits<double>::max();

    for (int j = 0; j < centroids.size(); ++j) {
        double currentDistSq = 0.0;

        for (size_t d = 0; d < numCols; ++d) {
            double diff = allData[rowNumber * numCols + d] - centroids[j][d];
            currentDistSq += diff * diff;
        }

        // Strikt kleiner: bij gelijke afstand wint de laagste index (zoals de referentie)
        if (currentDistSq < minDistanceSq) {
            minDistanceSq = currentDistSq;
            closestCluster = j;
        }
    }

    return std::make_pair(closestCluster, minDistanceSq);
}

// Gemiddelde van alle punten die tot cluster 'clusterIndex' horen
std::vector<double> average_of_points_with_cluster(const std::vector<std::vector<double>> &centroids, int clusterIndex, const std::vector<int> &clusters, const std::vector<double> &allData, size_t numCols)
{
    std::vector<double> sum(numCols, 0.0);
    size_t count = 0;

    for (size_t p = 0; p < clusters.size(); ++p) {
        if (clusters[p] == clusterIndex) {
            for (size_t d = 0; d < numCols; ++d)
                sum[d] += allData[p * numCols + d];
            ++count;
        }
    }

    // Lege cluster: niets om te middelen, dus de oude centroid behouden
    if (count == 0)
        return centroids[clusterIndex];

    for (size_t d = 0; d < numCols; ++d)
        sum[d] /= static_cast<double>(count);

    return sum;
}


int kmeans(Rng &rng, const std::string &inputFile, const std::string &outputFileName,
           int numClusters, int repetitions, int numBlocks, int numThreads,
           const std::string &centroidDebugFileName, const std::string &clusterDebugFileName)
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
	//////////////////////////////
	
	// This is a basic timer from std::chrono ; feel free to use the appropriate timer for
	// each of the technologies, e.g. OpenMP has omp_get_wtime()
	Timer timer;

	std::vector<int> bestClusters;
	double bestDistSquaredSum = std::numeric_limits<double>::max(); // can only get better

	std::vector<size_t> stepsPerRepetition(repetitions); // to save the number of steps each rep needed

    // Do the k-means routine a number of times, each time starting from
    // different random centroids (use Rng::pickRandomIndices), and keep 
    // the best result of these repetitions.
	for (int r = 0 ; r < repetitions ; r++)
	{
		size_t numSteps = 0;
        // TODO: perform an actual k-means run, starting from random centroids
        //       (see rng.h)
		std::cerr << "TODO: implement this" << std::endl;

		std::vector<size_t> indices(numClusters);		//K-means K = numClusters 	0 staat voor beginwaarde vector
		rng.pickRandomIndices(numRows, indices);			//#rows uit dataset = #datapunten  we krijgen dus vector terug met K random indices van de datapunten bv.indices = [42, 1089, 3500]


		// Kopieer de datapunten op deze indices naar een centroids-vector
        std::vector<std::vector<double>> centroids(numClusters);	// bv. [[1.2, 3.4][5.6, 7.8][2.1, 9.0]] k=3
				for (size_t j = 0; j < numClusters; ++j) {
					for (size_t d = 0; d < numCols; ++d) {
						centroids[j].push_back(allData[indices[j] * numCols + d]);
					}
				}

		std::vector<int> clusters(numRows, -1);	// Hier houden we bij welke kleur elk datapunt krijgt oftewel bij welke centroid het gaat behoren

		double distanceSquaredSum = 0;

		bool changed = true; 
		while (changed) {	// kijken per stap of de centroids nog veranderen
			changed = false;

			//clusters wegschrijven naar debug
			if (clustersDebugFile.is_open())
				clustersDebugFile.write(clusters);

			//centroids wegschrijven naar debug
			if (centroidDebugFile.is_open())
            	for (int j = 0; j < numClusters; j++)
					centroidDebugFile.write(centroids[j]);

					
			for (int p = 0; p < numRows; p++) {	//elk punt uit dataset afgaan
					std::pair<int, double> newCluster_distance = find_closest_centroid_index_and_distance(p, allData, numCols, centroids);
					distanceSquaredSum += newCluster_distance.second;

					if (newCluster_distance.first != clusters[p]) {	// als het punt bij een andere centroid hoort dan voorheen
						clusters[p] = newCluster_distance.first;	// update cluster index
						changed = true;	// er is een verandering, dus we moeten nog een stap doen
					}
			};

			// recalculate the centroids based on current clustering
			if (changed) {
				for (int j = 0; j < numClusters; j++){
					centroids[j] = average_of_points_with_cluster(centroids, j, clusters, allData, numCols);	// recalculeer de centroid van cluster j
				}
			}

			
			numSteps++;
		
		};

			// Keep track of best clustering
		if (distanceSquaredSum < bestDistSquaredSum) {
				bestDistSquaredSum = distanceSquaredSum;
				bestClusters = clusters;
			}

		stepsPerRepetition[r] = numSteps;

		// Make sure debug logging is only done on first iteration ; subsequent checks
		// with is_open will indicate that no logging needs to be done anymore.
		centroidDebugFile.close();
		clustersDebugFile.close();
	}

	timer.stop();

	// Some example output, of course you can log your timing data anyway you like.
	std::cerr << "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds" << std::endl;
	std::cout << "sequential," << numBlocks << "," << numThreads << "," << inputFile << "," 
			  << rng.getUsedSeed() << "," << numClusters << ","
		      << repetitions << "," << bestDistSquaredSum << "," << timer.durationNanoSeconds()/1e9
			  << std::endl;

	// Write the number of steps per repetition, kind of a signature of the work involved
	csvOutputFile.write(stepsPerRepetition, "# Steps: ");
	// Write best clusters to csvOutputFile, something like
	// csvOutputFile.write( best cluster indices )
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
	for (int i = 0 ; i < args.size() ; i += 2)
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
