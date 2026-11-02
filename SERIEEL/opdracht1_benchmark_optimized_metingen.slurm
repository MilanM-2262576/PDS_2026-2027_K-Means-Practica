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

# 1. Clean en recompile met optimalisatie (-O3 -DNDEBUG)
make -C src clean
make -C src

echo ""
echo "=== START 10 METINGEN ==="
echo ""

# 2. Voer de executable 10 keer uit 
for i in {1..10}
do
    echo "--- Run $i van 10 ---"
    ./src/kmeans --input datasets/100000x5.csv --output out.csv --k 10 --repetitions 20 --seed 1337
    echo ""
done

echo "=== EINDE METINGEN ==="