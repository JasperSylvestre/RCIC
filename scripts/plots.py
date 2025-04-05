"""
plots.py

Originally built in Google colab.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create necessary directories and enter scripts folder
if os.path.basename(os.getcwd()) != 'scripts':
  os.makedirs('scripts', exist_ok=True)
  os.chdir('scripts')
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../data/test/test_images', exist_ok=True)

# Comprehensive matrix to DataFrame
df = pd.DataFrame({
  'CPU speed': [0.5, 1.0, 1.6, 0.5, 1.0, 1.6, 0.5, 1.0, 1.6],
  'RAM': [512, 512, 512, 1024, 1024, 1024, 2048, 2048, 2048],
  'VGG16 prediction time': [1600, 950, 760, 710, 420, 280, 620, 340, 230],
  'ResNet50 prediction time': [400, 180, 130, 330, 170, 110, 340, 180, 120],
  'MobileNetV2 prediction time': [76, 39, 26, 69, 38, 26, 71, 43, 27],
  'Pre-processing time': [4.1, 2.6, 1.9, 3.8, 2.1, 1.6, 3.1, 1.5, 1.1]
})

# Define plot parameters
colors = plt.cm.plasma(np.linspace(0, 0.5, 3))
markers = ['o', 's', 'D']

cpu_speeds = df['CPU speed'].unique()
ram_values = df['RAM'].unique()
models = ['VGG16', 'ResNet50', 'MobileNetV2']

plt.figure(figsize=(10, 6))
plt.title('Prediction Time vs CPU Speed by RAM')
plt.xlabel('CPU Speed (GHz)')
plt.xticks(cpu_speeds)
plt.ylabel('Prediction Time (s)')
plt.yscale('log')
plt.tight_layout()

# Plot prediction time for different models
for i, model in enumerate(models):
  for j, ram in enumerate(ram_values):
    df_ram = df[df['RAM'] == ram]
    pred_time = df_ram[f'{model} prediction time']
    ls = ':' if ram == 512 and model in ['VGG16', 'ResNet50'] else '-'  # dotted line to indicate swap memory
    plt.plot(cpu_speeds, pred_time, marker=markers[j], markersize=7, markerfacecolor='none', markeredgewidth=1, linestyle=ls, color=colors[i])

# Increase ylim
ymin, ymax = plt.ylim()
plt.ylim(ymin / 1.5, ymax * 1.5)

# Create legends
legends = []
for i, model in enumerate(models):
  legends.append(plt.Line2D([], [], color=colors[i], label=model, linestyle='-', marker='None'))        # model

for i, ram in enumerate(ram_values):
  legends.append(plt.Line2D([], [], color='black', label=str(ram), linestyle='-', marker=markers[i]))   # ram values

legends.append(plt.Line2D([], [], color='black', label='Swap used', linestyle=':', marker='None'))      # swap

plt.legend(handles=legends, loc='upper right', bbox_to_anchor=(1.21, 1))

# Save/show
plt.savefig('../results/prediction_time_vs_cpu_speed_by_ram_plot.png', bbox_inches='tight')
plt.show()

df = df[['CPU speed', 'RAM', 'Pre-processing time']]

# Define plot parameters
plt.figure(figsize=(10, 6))
plt.title('Pre-processing Time vs CPU Speed by RAM')
plt.xlabel('CPU Speed (GHz)')
plt.xticks(cpu_speeds)
plt.ylabel('Pre-processing Time (s)')
plt.tight_layout()

# Plot pre-processing times
for i, ram in enumerate(ram_values):
  ram_df = df[df['RAM'] == ram]
  plt.plot(ram_df['CPU speed'], ram_df['Pre-processing time'], label=f'{ram} MB RAM', marker='o', color=colors[i])

# Increase ylim
ymin, ymax = plt.ylim()
plt.ylim(0, ymax * 1.1)

plt.legend()

# Save/show
plt.savefig('../results/pre-processing_time_vs_cpu_speed_by_ram_plot.png')
plt.show()
