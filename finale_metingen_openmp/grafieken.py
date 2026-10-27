import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Jouw data inlezen uit de gegeven uitvoer
data_str = """Type,blocks,threads,file,seed,clusters,repetitions,bestdistsquared,timeinseconds
sequential,1,1,../datasets/100000x5.csv,1338,10,20,2.25117e+12,3.91757
sequential,1,1,../datasets/100000x5.csv,1338,10,20,2.25117e+12,3.91351
sequential,1,1,../datasets/100000x5.csv,1338,10,20,2.25117e+12,3.92349
sequential,1,1,../datasets/100000x5.csv,1338,10,20,2.25117e+12,3.91477
sequential,1,1,../datasets/100000x5.csv,1338,10,20,2.25117e+12,3.91393
sequential,1,2,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.98901
sequential,1,2,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.99455
sequential,1,2,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.99159
sequential,1,2,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.99311
sequential,1,2,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.99178
sequential,1,4,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.01496
sequential,1,4,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.02832
sequential,1,4,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.02511
sequential,1,4,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.02365
sequential,1,4,../datasets/100000x5.csv,1338,10,20,2.25117e+12,1.02894
sequential,1,8,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.719434
sequential,1,8,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.72014
sequential,1,8,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.720622
sequential,1,8,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.720319
sequential,1,8,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.719797
sequential,1,16,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.626766
sequential,1,16,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.628834
sequential,1,16,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.629394
sequential,1,16,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.625635
sequential,1,16,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.626183
sequential,1,32,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.630349
sequential,1,32,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.629563
sequential,1,32,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.630188
sequential,1,32,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.628028
sequential,1,32,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.626528
sequential,1,64,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.637454
sequential,1,64,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.637758
sequential,1,64,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.64428
sequential,1,64,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.640536
sequential,1,64,../datasets/100000x5.csv,1338,10,20,2.25117e+12,0.63878"""

# Als je een CSV-bestand gebruikt, gebruik je: df = pd.read_csv('jouw_bestand.csv')
df = pd.read_csv(io.StringIO(data_str))

# 2. Gemiddelde en standaarddeviatie berekenen per aantal threads
grouped = df.groupby("threads")["timeinseconds"].agg(["mean", "std"]).reset_index()

threads = grouped["threads"].values
mean_time = grouped["mean"].values
std_time = grouped["std"].fillna(0).values  # Standaarddeviatie voor errorbars

# 3. Berekening van Speedup en Efficientie
# T1 is de gemiddelde tijd voor 1 thread
t1 = mean_time[0]

speedup = t1 / mean_time
# Foutverplanting voor speedup: std(S) ≈ S * (std(T) / T)
speedup_std = speedup * (std_time / mean_time)

efficiency = speedup / threads
efficiency_std = speedup_std / threads

# 4. Grafieken tekenen (3 subplots naast elkaar)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Grafiek 1: Uitvoertijd ---
axes[0].errorbar(
    threads,
    mean_time,
    yerr=std_time,
    fmt="-o",
    markersize=3,
    capsize=4,
    color="tab:blue",
)
axes[0].set_xlabel("Aantal threads")
axes[0].set_ylabel("Uitvoertijd (s)")
axes[0].set_ylim(bottom=0)

# --- Grafiek 2: Speedup factor ---
axes[1].plot(
    threads, threads, label="Maximale speedup", color="tab:orange", linestyle="-"
)
axes[1].errorbar(
    threads,
    speedup,
    yerr=speedup_std,
    fmt="-o",
    markersize=3,
    capsize=4,
    color="tab:blue",
    label="Gemeten speedup",
)
axes[1].set_xlabel("Aantal threads")
axes[1].set_ylabel("Speedup factor")
axes[1].legend()

# --- Grafiek 3: Efficientie ---
axes[2].errorbar(
    threads,
    efficiency,
    yerr=efficiency_std,
    fmt="-o",
    markersize=3,
    capsize=4,
    color="tab:blue",
)
axes[2].set_xlabel("Aantal threads")
axes[2].set_ylabel("Efficientie")
axes[2].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()