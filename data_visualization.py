import pandas as pd
import plotly.express as px
import numpy as np
import argparse
import os
from constants import MODELS, FOODS, NICKNAMES

# Args.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='FoodSeg103', help='dataset')
args = parser.parse_args()
dataset = args.dataset
form = 'svg'

if not os.path.exists(f'data/{dataset}/Images/inf_dir/stored_metrics.csv'):
    raise ValueError(f"File not found: data/{dataset}/Images/inf_dir/stored_metrics.csv")
os.makedirs('data/graphs', exist_ok=True)

# Organize data.
# Data Format: ["time", "model_name", "threshold", "nms", "poly", "topk", "mIoU", "mAcc", "aAcc"]
df = pd.read_csv(f'data/{dataset}/Images/inf_dir/stored_metrics.csv', 
                  parse_dates=['time'],
                  dtype={'time': str, 'model_name': str, 'threshold': float, 'nms': float, 'poly': str, 'topk': float, 'mIoU': float, 'mAcc': float, 'aAcc': float})
df['model_name'] = df['model_name'].map(NICKNAMES)
df['topk'] = df['topk'].fillna("None")
df.rename(columns={'topk': 'top-k', 'model_name': 'model'}, inplace=True)
df = (
    df.sort_values(by='time')
    .groupby(['model', 'threshold', 'nms', 'poly', 'top-k'], as_index=False)
    .last()
)
df.sort_values(by=['model', 'mIoU'], ascending=[True, False], inplace=True)

# Filter data.
# By Model
OWLv2_Base = (df['model'] == 'OWLv2 Base')
OWLv2_Large = (df['model'] == 'OWLv2 Large')
OWLViT_Base = (df['model'] == 'OWLViT Base')
OWLViT_Large = (df['model'] == 'OWLViT Large')
OmDet = (df['model'] == 'OmDet')

# Top K
AllTopk = (df['top-k'] != 'None')
NoTopk = (df['top-k'] == 'None')
FixTopk = (df['top-k'] == 6)

# Threshold
AllThreshold = (df['threshold'] != 0)
NoThreshold = (df['threshold'] == 0)
FixThreshold = (df['threshold'] == 0.4)

# Poly
YesPoly = (df['poly'] == 'True')
NoPoly = (df['poly'] == 'False')

# Nms
AllNms = (df['nms'] != 0)
FixNms = (df['nms'] == .3)

# Hover data
hover_data = ['threshold', 'top-k', 'poly']


# PLOT: OWLv2 Base Top-K Metrics
models = df[OWLv2_Base & AllTopk & NoThreshold & YesPoly & FixNms].sort_values(by='top-k')
fig = px.line(models, x='top-k', y=['mIoU', 'mAcc', 'aAcc'], title='OWLv2 Base: Top-K Metrics', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/OWLv2_Base_Top-K_Metrics.{form}")

# PLOT: OWLv2 Large Top-K Metrics
models = df[OWLv2_Large & AllTopk & NoThreshold & YesPoly & FixNms].sort_values(by='top-k')
fig = px.line(models, x='top-k', y=['mIoU', 'mAcc', 'aAcc'], title='OWLv2 Large: Top-K Metrics', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/OWLv2_Large_Top-K_Metrics.{form}")

# PLOT: OWLv2 Large Threshold Metrics
models = df[OWLv2_Large & AllThreshold & NoTopk & YesPoly & FixNms].sort_values(by='threshold')
fig = px.line(models, x='threshold', y=['mIoU', 'mAcc', 'aAcc'], title='OWLv2 Large: Threshold Metrics', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/OWLv2_Large_Threshold_Metrics.{form}")

# PLOT: All Models Threshold Metrics
models = df[AllThreshold & FixThreshold & YesPoly & FixNms]
fig = px.bar(models, x='model', y='mIoU', title='mIoU by Model (.4 threshold)', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/All_Models_Threshold_Metrics.{form}")

# PLOT: All Models Top-K Metrics
models = df[AllTopk & FixTopk & YesPoly & FixNms]
fig = px.bar(models, x='model', y='mIoU', title='mIoU by Model (top-6)', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/All_Models_Top-K_Metrics.{form}")

# PLOT: OmDet Nms Metrics
models = df[OmDet & AllNms & YesPoly & FixThreshold & NoTopk].sort_values(by='nms')
fig = px.line(models, x='nms', y=['mIoU', 'mAcc', 'aAcc'], title='OmDet: Nms Metrics', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/OmDet_Nms_Metrics.{form}")

# PLOT: OWLv2 Large Poly Metrics
models_y = df[OWLv2_Large & AllThreshold & NoTopk & FixNms].sort_values(by='threshold')
fig = px.line(models_y, x='threshold', y='mIoU', title='OWLv2 Large: Poly Metrics', color='poly', hover_data=hover_data)
fig.show()
fig.write_image(f"data/graphs/OWLv2_Large_Poly_Metrics.{form}")

# Plot all mIoU for all models (scatter plot)
fig = px.scatter(df, x='model', y='mIoU', title='mIoU by Model', hover_data=hover_data, color='model')
fig.show()
fig.write_image(f"data/graphs/All_Models_mIoU.{form}")