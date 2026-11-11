#!/bin/bash -l
#SBATCH --cluster=mindwell
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:05:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

# Navigeer naar de src map waar de bronbestanden en Makefile staan
cd ~/OPENMP_PRACTICA/src

module --force purge
module load cluster/mindwell/batch_graniterapids
module load GCC/14.3.0

g++ --version

# Clean & Build
make -f Makefile.openmp clean
make -f Makefile.openmp

INPUT="../datasets/100000x5.csv"
K=10
REPS=200
SEED=2262576
NRUNS=5

echo "=== START KMEANS OPENMP ==="
echo "# Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds"

for T in 1 2 4 8 16 32 64 128 144 192; do
    for RUN in $(seq 1 $NRUNS); do
        ./kmeans_omp --input $INPUT --output out_t${T}_run${RUN}.csv \
            --k $K --repetitions $REPS --seed $SEED --threads $T
    done
done

echo "=== UITVOERING VOLTOOID ==="