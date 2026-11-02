#!/bin/bash -l
#SBATCH --cluster=mindwell
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=00:05:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module --force purge
module load cluster/mindwell/batch_graniterapids
module load GCC/13.3.0

gcc --version 2>&1

echo "=================================================="
echo "          1. TEST MET OPTIMALISATIE  -O3           "
echo "=================================================="

make -C src clean
make -C src 

echo " --- Start meting --- "
python src/compare.py src/kmeans /data/leuven/303/vsc30380/kmeans_serial_reference --input datasets/100000x5.csv --output out.csv --k 10 --repetitions 20 --seed 1337 --trace clustertrace.csv --centroidtrace centroidtrace.csv

