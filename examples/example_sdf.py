"""
Signed Distance Function (SDF)
==========================
This module provides the SDF class to train a neural network model for the signed distance function from URDF/MJCF files and visualize the reconstructed whole body.
"""
import argparse
import os

import numpy as np
import torch

import robolab

# Initialize arguments
args = argparse.Namespace(
    numFuncs=8,
    domainMin=-1.0,
    domainMax=1.0,
    device='cuda',
    assetName='wheel_chair',
    assetRoot='../assets',
    assetFile='mjcf/wheel_chair/wheel_chair.xml',
    trainEpochs=1000,
    saveMeshDict=True
)

# Initialize the SDF class
sdf_model = robolab.wdf.SDF(args)

# Train the SDF model
asset_path = os.path.join(args.assetRoot, args.assetFile)
model_path = os.path.join(os.path.dirname(asset_path), "sdf", "BP", f"BP_{args.numFuncs}.pt")
if not os.path.exists(model_path):
    sdf_model.train()

# Generate surface mesh
trained_model = torch.load(model_path)
sdf_model.create_surface_mesh(trained_model, nbData=128, vis=True, save_mesh_name=f'BP_{args.numFuncs}')

# Generate random points and get the SDF and gradients
points = np.random.rand(100, 3) * 2.0 - 1.0
sdf, gradient = sdf_model.get_sdf_batch(points, trained_model, base_trans=None, use_derivative=True)
