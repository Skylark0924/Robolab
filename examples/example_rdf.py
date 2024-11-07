"""
Robot distance field (RDF)
========================

This example demonstrates how to use the RDF class to train a Bernstein Polynomial model for the robot distance field
from URDF/MJCF files and visualize the reconstructed whole body.
"""

import argparse
import os
import time

import numpy as np
import torch

import robolab


def rdf_from_robot_model(args):
    rdf_bp = robolab.wdf.RDF(args, robot_verbose=True)
    asset_path = os.path.join(args.assetRoot, args.assetFile)
    rdf_dir = os.path.join(os.path.dirname(asset_path), "rdf")

    #  train Bernstein Polynomial model
    rdf_model_path = os.path.join(rdf_dir, 'BP', f'BP_{args.numFuncs}.pt')
    if not os.path.exists(rdf_model_path) or args.forceTrain:  # train the model
        rdf_bp.train()
    rdf_model = torch.load(rdf_model_path)

    # visualize the Bernstein Polynomial model for each robot link
    rdf_bp.create_surface_mesh(rdf_model, nbData=128, vis=False, save_mesh_name=f'BP_{args.numFuncs}')

    num_joint = rdf_bp.robot.num_joint
    joint_value = torch.zeros(num_joint).to(args.device)

    base_trans = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().to(args.device)
    trans_dict = rdf_bp.robot.get_trans_dict(joint_value, base_trans)
    # visualize the Bernstein Polynomial model for the whole body
    rdf_bp.visualize_reconstructed_whole_body(rdf_model, trans_dict, tag=f'BP_{args.numFuncs}')

    # run RDF
    x = torch.rand(10, 3).to(args.device) * 2.0 - 1.0
    joint_value = torch.rand(100, rdf_bp.robot.num_joint).to(args.device).float()
    base_trans = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(joint_value), 4,
                                                                                           4).float().to(args.device)

    start_time = time.time()
    sdf, gradient = rdf_bp.get_whole_body_sdf_batch(x, joint_value, rdf_model, base_trans=base_trans,
                                                    use_derivative=True)
    print('Time cost:', time.time() - start_time)
    print('sdf:', sdf.shape, 'gradient:', gradient.shape)

    start_time = time.time()
    sdf, joint_grad = rdf_bp.get_whole_body_sdf_with_joints_grad_batch(x, joint_value, rdf_model, base_trans=base_trans)
    print('Time cost:', time.time() - start_time)
    print('sdf:', sdf.shape, 'joint gradient:', joint_grad.shape)

    # visualize the 2D & 3D SDF with gradient
    joint_value = torch.zeros(num_joint).to(args.device).reshape((-1, num_joint))
    # joint_value = torch.rand(num_joint).to(args.device).reshape((-1, num_joint)) * (joint_max - joint_min) + joint_min

    # joint_value = (torch.rand(num_joint).to(args.device).reshape((-1, num_joint))*0.5 * (joint_max - joint_min) + joint_min)
    # robolab.rdf.plot_2D_sdf(joint_value, rdf_bp, nbData=80, model=rdf_model, device=args.device)
    robolab.wdf.plot_3D_sdf_with_gradient(joint_value, rdf_bp, model=rdf_model, device=args.device)
    #    used_links=["shell", "head_link1", "head_link2", "panda_left_link1",
    #                "panda_left_link2", "panda_left_link3", "panda_left_link4",
    #                "panda_left_link5", "panda_left_link6", "panda_left_link7",
    #                "panda_right_link1", "panda_right_link2",
    #                "panda_right_link3", "panda_right_link4",
    #                "panda_right_link5", "panda_right_link6",
    #                "panda_right_link7"])
    # used_links=["shell"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domainMax', default=1.0, type=float)
    parser.add_argument('--domainMin', default=-1.0, type=float)
    parser.add_argument('--numFuncs', default=16, type=int)
    parser.add_argument('--trainEpochs', default=1000, type=int)
    parser.add_argument('--forceTrain', action='store_true')
    parser.add_argument('--saveMeshDict', action='store_false')
    parser.add_argument('--samplePoints', action='store_false')
    parser.add_argument('--parallel', action='store_false')
    parser.add_argument('--assetName', default="Bruce", type=str)
    parser.add_argument('--assetRoot', default="../assets", type=str)
    parser.add_argument('--assetFile', default="mjcf/bruce/bruce.xml", type=str)
    parser.add_argument('--baseLink', default="pelvis", type=str)
    args = parser.parse_args()

    rdf_from_robot_model(args)
