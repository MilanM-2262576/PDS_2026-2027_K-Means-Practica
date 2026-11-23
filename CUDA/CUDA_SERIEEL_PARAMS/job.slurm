#!/bin/bash -l
#SBATCH --cluster=mindwell
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module --force purge
module load cluster/mindwell/batch_graniterapids
module load GCC/14.3.0

g++ --version

# 1. Clean en recompile met optimalisatie (-O3 -DNDEBUG)
make -C ../../src -f Makefile clean
make -C ../../src -f Makefile

echo ""
echo "=== START 3 METINGEN ==="
echo ""

# 2. Voer de executable 3 keer uit 
for i in {1..5}
do
    echo "--- Run $i van 3 ---"
    ../../src/kmeans --input ../../datasets/1M_1000000x4.csv --output out_serial_cuda_1M_$i.csv --k 15 --repetitions 10 --seed 2262576
    echo ""
done

echo "=== EINDE METINGEN ==="