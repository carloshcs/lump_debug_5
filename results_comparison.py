# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:51:53 2025

@author: saunders
"""

import pandas as pd
import matplotlib.pyplot as plt

# File paths
abq_lump_path = "abq_lump1.txt"
substrate_path = "./results/Lump_100_dt_05.txt"
substrate_path_conv = "./results/Lump_100_08_06b_07l.txt"
substrate_4 = "./results/Lump_100_08_06b_1l.txt"


# Parameters
use_mean = False   # False = use T_point_degC, True = use T_mean_degC
max_layer = 50     # Change this value to limit x-axis by layer

# Read files (assuming whitespace/tab separated, ignoring # comments)
abq_df = pd.read_csv(abq_lump_path, sep=r"\s+",
                     comment="#", header=None, names=["Layer", "Time", "Temperature"])
substrate_df = pd.read_csv(substrate_path, sep=r"\s+",
                           comment="#", header=None, names=["Layer", "T_point_degC", "T_mean_degC"])

substrate_path_conv_df = pd.read_csv(substrate_path_conv, sep=r"\s+",
                           comment="#", header=None, names=["Layer", "T_point_degC", "T_mean_degC"])

substrate_4 = pd.read_csv(substrate_path_conv, sep=r"\s+",
                           comment="#", header=None, names=["Layer", "T_point_degC", "T_mean_degC"])



# Pick Abaqus column (point or mean)
substrate_temp_col = "T_mean_degC" if use_mean else "T_point_degC"

# Filter by max_layer
abq_df = abq_df[abq_df["Layer"] <= max_layer]
substrate_df = substrate_df[substrate_df["Layer"] <= max_layer]

# Plot
plt.figure(figsize=(8,5))
plt.plot(abq_df["Layer"], abq_df["Temperature"], marker="o", label= f"Abaqus method ({substrate_temp_col})")
plt.plot(substrate_df["Layer"], substrate_df[substrate_temp_col], marker="s", 
         label="Lump_100_dt_05")
plt.plot(substrate_path_conv_df["Layer"], substrate_path_conv_df[substrate_temp_col], marker="x", 
         label="Lump_1_dt_008")

plt.plot(substrate_4["Layer"], substrate_4[substrate_temp_col], marker="x", 
         label="substrate_4")

plt.xlabel("Layer")
plt.ylabel("Temperature (°C)")
plt.title("Temperature vs Layer — Abaqus method vs Lump method")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
