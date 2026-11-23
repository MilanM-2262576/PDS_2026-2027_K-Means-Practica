#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=llp_h_pds
#SBATCH --time=00:15:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module purge
module load GCC/10.3.0
module load CUDA/11.7.0

nvcc -O3 -DNDEBUG -o kmeans_cuda ../src/main_startcode_cuda_no_managed_mem_gem_aan.cu ../src/rng.cpp -gencode arch=compute_80,code=sm_80

THREADS=512
for BLOCKS in 2 4 8 16 32 64 108 216 432 864; do
    for RUN in {1..3}; do
        echo "BLOCKS=$BLOCKS RUN=$RUN"
        ./kmeans_cuda --input ../datasets/1M_1000000x4.csv --output out_t${BLOCKS}_run${RUN}.csv --k 15 --repetitions 10 --seed 2262576 --blocks $BLOCKS --threads $THREADS
    done
done
