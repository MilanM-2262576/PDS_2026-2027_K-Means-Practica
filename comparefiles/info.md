output compare commando:

        bash-5.1$ python compare.py kmeans /data/leuven/303/vsc30380/kmeans_serial_reference --input ../datasets/mouse_500x2.csv --output out.csv --k 3 --repetitions 10 --seed 12345 --trace clustertrace.csv --centroidtrace centroidtrace.csv 
        # Running ['kmeans', '--input', '../datasets/mouse_500x2.csv', '--k', '3', '--output', 'out.csv.1', '--repetitions', '10', '--seed', '12345', '--trace', 'clustertrace.csv.1', '--centroidtrace', 'centroidtrace.csv.1']
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        TODO: implement this
        # Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds
        sequential,1,1,../datasets/mouse_500x2.csv,12345,3,10,8.11316,0.00219111
        # Running ['/data/leuven/303/vsc30380/kmeans_serial_reference', '--input', '../datasets/mouse_500x2.csv', '--k', '3', '--output', 'out.csv.2', '--repetitions', '10', '--seed', '12345', '--trace', 'clustertrace.csv.2','--centroidtrace', 'centroidtrace.csv.2']
        # Loaded 500x2 matrix
        # New best on repetition 0: 8.11316 (difference is 1.79769e+308)
        # Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds
        sequential_icc,1,1,../datasets/mouse_500x2.csv,12345,3,10,8.11316,0.00184643
        # Output checks out: ['out.csv.1', 'out.csv.2']
        # Output checks out: ['clustertrace.csv.1', 'clustertrace.csv.2']
        # Output checks out: ['centroidtrace.csv.1', 'centroidtrace.csv.2']
        # Done