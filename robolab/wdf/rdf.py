#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

import sys

sys.setrecursionlimit(100000)
import os

import numpy as np

np.set_printoptions(threshold=np.inf)
import trimesh
import mesh_to_sdf
import skimage
from tqdm import tqdm
import robolab
from robolab.wdf import utils
from robolab.wdf.simple_shape_sdf import *
from robolab.utils import create_dir


class RDF:
    def __init__(self, args, robot_verbose=False):
        """
        Use Bernstein Polynomial to represent the SDF of the robot
        """
        self.args = args
        self.n_func = args.numFuncs
        self.domain_min = args.domainMin
        self.domain_max = args.domainMax
        self.device = args.device
        self.asset_path = os.path.join(args.assetRoot, args.assetFile)
        self.wdf_dir = os.path.join(os.path.dirname(self.asset_path), "rdf")
        self.save_mesh_dict = args.saveMeshDict
        self.simple_shape = False  # Forced to be False

        # Build the robot from the URDF/MJCF file
        self.robot = robolab.RobotModel(self.asset_path, solve_engine="pytorch_kinematics", device=self.device,
                                        verbose=robot_verbose, base_link=args.baseLink)
        assert os.path.exists(self.robot.mesh_dir), "Please organize the robot meshes in the 'meshes' folder!"

        self.link_list = self.robot.get_link_list()
        self.link_mesh_map = self.robot.link_mesh_map
        self.link_meshname_map = self.robot.link_meshname_map

    def _binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def _build_bernstein_t(self, t, use_derivative=False):
        # t is normalized to [0,1]
        t = torch.clamp(t, min=1e-4, max=1 - 1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self._binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(), None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (
                    1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(), dphi.float()

    def _build_basis_function_from_points(self, points, use_derivative=False):
        N = len(points)
        points = ((points - self.domain_min) / (self.domain_max - self.domain_min)).reshape(-1)
        phi, d_phi = self._build_bernstein_t(points, use_derivative)
        phi = phi.reshape(N, 3, self.n_func)
        phi_x = phi[:, 0, :]
        phi_y = phi[:, 1, :]
        phi_z = phi[:, 2, :]
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func ** 2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func ** 3)
        if not use_derivative:
            return phi_xyz, None
        else:
            d_phi = d_phi.reshape(N, 3, self.n_func)
            d_phi_x_1D = d_phi[:, 0, :]
            d_phi_y_1D = d_phi[:, 1, :]
            d_phi_z_1D = d_phi[:, 2, :]
            d_phi_x = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", d_phi_x_1D, phi_y).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_y = torch.einsum("ij,ik->ijk",
                                   torch.einsum("ij,ik->ijk", phi_x, d_phi_y_1D).view(-1, self.n_func ** 2),
                                   phi_z).view(-1, self.n_func ** 3)
            d_phi_z = torch.einsum("ij,ik->ijk", phi_xy, d_phi_z_1D).view(-1, self.n_func ** 3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1), d_phi_y.unsqueeze(-1), d_phi_z.unsqueeze(-1)), dim=-1)
            return phi_xyz, d_phi_xyz

    def train(self):
        # 初始化用于保存训练结果的字典
        mesh_dict = {}

        # sample points for each mesh
        if self.args.samplePoints:
            save_path = os.path.join(self.wdf_dir, 'sdf_points')
            create_dir(save_path)

            if self.args.parallel:
                import multiprocessing
                multiprocessing.set_start_method('spawn')
                pool = multiprocessing.Pool(processes=12)  # 可以根据硬件调整进程数

                # 构建任务列表 (只处理尚未采样的几何体)
                task_list = []
                for link_name, geom_dict in self.link_mesh_map.items():
                    for geom_name, geom_info in geom_dict.items():
                        # 检查是否已经采样过该几何体
                        save_file = os.path.join(save_path,
                                                 f'voxel_128_{link_name}_{geom_name}_{geom_info.get("type")}.npy')
                        if not os.path.exists(save_file):
                            task_list.append((link_name, geom_name, save_path, geom_info))

                # 分发任务并采集结果
                data_list = pool.map(job, task_list)

                # 保存所有采样结果
                for result in data_list:
                    link_name = result['link_name']
                    geom_name = result['mesh_name']
                    np.save(os.path.join(save_path, f'voxel_128_{link_name}_{geom_name}_{geom_info.get("type")}.npy'),
                            result)

                pool.close()
                pool.join()
            else:
                # 遍历 self.link_mesh_map 中的 link 和 geom
                for link_name, geom_dict in self.link_mesh_map.items():
                    for geom_name, geom_info in geom_dict.items():
                        # 检查是否已经采样过该几何体
                        if os.path.exists(os.path.join(save_path,
                                                       f'voxel_128_{link_name}_{geom_name}_{geom_info.get("type")}.npy')):
                            continue  # 如果已经采样过，跳过
                        data = sample_sdf_points(link_name, geom_name, save_path, geom_info)

                        # 保存采样的 SDF 数据
                        np.save(
                            os.path.join(save_path, f'voxel_128_{link_name}_{geom_name}_{geom_info.get("type")}.npy'),
                            data)

        # 定义训练单个几何体的函数
        def train_single_mesh(link_name, mesh_name, data):
            """
            训练单个几何体（可能是简单几何体或复杂 mesh）

            :param link_name: Link 名称
            :param mesh_name: 几何体名称
            :param data: 采样的 SDF 数据
            :param is_simple_shape: 是否为简单几何体
            :return: 包含训练结果的字典
            """
            # 对于 mesh，使用采样的 SDF 数据
            center = data['center']
            scale = data['scale']
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data < -1] = -sdf_random_data[sdf_random_data < -1]
            wb = torch.zeros(self.n_func ** 3).float().to(self.device)
            batch_size = (torch.eye(self.n_func ** 3) / 1e-4).float().to(self.device)

            for iter in range(self.args.trainEpochs):
                choice_near = np.random.choice(len(point_near_data), 1024, replace=False)
                p_near, sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(
                    self.device), torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)

                choice_random = np.random.choice(len(point_random_data), 256, replace=False)
                p_random, sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(
                    self.device), torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)

                p = torch.cat([p_near, p_random], dim=0)
                sdf = torch.cat([sdf_near, sdf_random], dim=0)
                phi_xyz, _ = self._build_basis_function_from_points(p.float().to(self.device), use_derivative=False)

                K = torch.matmul(batch_size, phi_xyz.T).matmul(torch.linalg.inv(
                    (torch.eye(len(p)).float().to(self.device) + torch.matmul(torch.matmul(phi_xyz, batch_size),
                                                                              phi_xyz.T))))
                batch_size -= torch.matmul(K, phi_xyz).matmul(batch_size)
                delta_wb = torch.matmul(K, (sdf - torch.matmul(phi_xyz, wb)).squeeze())
                wb += delta_wb

            return {
                'link_name': link_name,
                'mesh_name': mesh_name,
                'weights': wb,
                'offset': torch.from_numpy(center),
                'scale': scale,
            }

        with tqdm(self.link_mesh_map.items(), desc="Training") as pbar:
            for link_name, geom_dict in pbar:
                for geom_name, geom_info in geom_dict.items():
                    # 加载采样的 SDF 数据
                    sampled_point_data = np.load(
                        f'{self.wdf_dir}/sdf_points/voxel_128_{link_name}_{geom_name}_{geom_info.get("type")}.npy',
                        allow_pickle=True).item()
                    res = train_single_mesh(link_name, geom_name, sampled_point_data)
                    mesh_dict[f'{geom_name}'] = res
                    pbar.set_postfix({'Finished': f'{link_name} - {geom_name}'})

        self.mesh_dict = mesh_dict

        if self.save_mesh_dict:
            rdf_model_path = os.path.join(self.wdf_dir, 'BP')
            create_dir(rdf_model_path)
            torch.save(mesh_dict, f'{rdf_model_path}/BP_{self.n_func}.pt')
            print(f'{rdf_model_path}/BP_{self.n_func}.pt model saved!')

    def sdf_to_mesh(self, model, nbData, use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].to(self.device)

            domain = torch.linspace(self.domain_min, self.domain_max, nbData).to(self.device)
            grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
            p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(self.device)

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d = []
            for p_s in p_split:
                phi_p, d_phi_p = self._build_basis_function_from_points(p_s, use_derivative)
                d_s = torch.matmul(phi_p, weights)
                d.append(d_s)
            d = torch.cat(d, dim=0)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0,
                spacing=np.array([(self.domain_max - self.domain_min) / nbData] * 3)
            )
            verts = verts - [1, 1, 1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list, mesh_name_list

    def create_surface_mesh(self, model, nbData, vis=False, save_mesh_name=None):
        verts_list, faces_list, mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces, mesh_name in zip(verts_list, faces_list, mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts, faces)
            if vis:
                print(f"Visualizing {mesh_name}")
                rec_mesh.show()
            if save_mesh_name is not None:
                save_path = os.path.join(self.wdf_dir, "output_meshes")
                create_dir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh,
                                                    os.path.join(save_path, f"{save_mesh_name}_{mesh_name}.stl"))

    def get_whole_body_sdf_batch(self, points, joint_value, model, base_trans=None, use_derivative=True,
                                 used_links=None):
        B = len(joint_value)  # batch size
        N = len(points)

        if used_links is None:
            used_links = self.robot.real_link
            used_links = [link for link in used_links if link in self.link_mesh_map]

        offset_list = []
        scale_list = []
        weights_list = []
        trans_list = []
        trans_dict = self.robot.get_trans_dict(joint_value, base_trans)
        index_list = []
        for used_link in used_links:
            if used_link in self.link_mesh_map:
                mesh_names = self.link_mesh_map[used_link]
                for mesh_name in mesh_names:
                    index = list(self.link_meshname_map.keys()).index(used_link)
                    offset = model[mesh_name]['offset'].unsqueeze(0)
                    scale = model[mesh_name]['scale']
                    weights = model[mesh_name]['weights'].unsqueeze(0)

                    index_list.append(index)
                    offset_list.append(offset)
                    scale_list.append(scale)
                    weights_list.append(weights)
                    trans = trans_dict[used_link]
                    trans_list.append(trans)

        K = len(offset_list)
        offset = torch.cat(offset_list, dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B * K, 3).float()
        scale = torch.tensor(scale_list, device=self.device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B * K).float()
        weights_near = torch.cat(weights_list, dim=0).to(self.device).float()
        trans = torch.cat(trans_list, dim=0).to(self.device).float()
        trans = trans.reshape(K, B, 4, 4)

        fk_trans = torch.cat([t.unsqueeze(1) for t in trans], dim=1).reshape(-1, 4, 4)  # B,K,4,4
        x_robot_frame_batch = utils.transform_points(points.float(), torch.linalg.inv(fk_trans).float(),
                                                     device=self.device)  # B*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)  # B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled > 1.0 - 1e-2, 1.0 - 1e-2, x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded < -1.0 + 1e-2, -1.0 + 1e-2, x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi, _ = self._build_basis_function_from_points(x_bounded.reshape(B * K * N, 3),
                                                            use_derivative=False)
            phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B * N, -1)  # K,B*N,-1

            # sdf
            sdf = torch.einsum('ijk,ik->ij', phi, weights_near).reshape(K, B, N).transpose(0, 1).reshape(
                B * K, N)  # B,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            return sdf_value, None
        else:
            phi, dphi = self._build_basis_function_from_points(x_bounded.reshape(B * K * N, 3),
                                                               use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1), dphi], dim=-1)
            phi_cat = phi_cat.reshape(B, K, N, -1, 4).transpose(0, 1).reshape(K, B * N, -1,
                                                                              4)  # K,B*N,-1,4

            output = torch.einsum('ijkl,ik->ijl', phi_cat, weights_near).reshape(K, B, N, 4).transpose(0,
                                                                                                       1).reshape(
                B * K, N, 4)
            sdf = output[:, :, 0]
            gradient = output[:, :, 1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B, K, N)
            sdf = sdf * (scale.reshape(B, K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient, dim=-1)
            gradient = torch.nn.functional.normalize(gradient, dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = fk_trans[:, :3, :3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl', fk_rotation, gradient.transpose(1, 2)).transpose(1,
                                                                                                                2).reshape(
                B, K, N, 3)
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx = idx.unsqueeze(1).unsqueeze(-1).expand(B, K, N, 3)
            gradient_value = torch.gather(gradient_base_frame, 1, idx)[:, 0, :, :]
            # gradient_value = None
            return sdf_value, gradient_value

    def get_whole_body_sdf_with_joints_grad_batch(self, points, joint_value, model, base_trans=None, used_links=None):
        """
        Get the SDF value and gradient of the whole body with respect to the joints

        :param points: (batch_size, 3)
        :param joint_value: (batch_size, joint_num)
        :param model: the trained RDF model
        :param base_trans: the transformation matrix of base pose, (1, 4, 4)
        :param used_links: the links to be used, list of link names
        :return:
        """
        delta = 0.001
        batch_size = joint_value.shape[0]
        joint_num = joint_value.shape[1]
        link_num = len(self.robot.get_link_list())
        joint_value = joint_value.unsqueeze(1)

        d_joint_value = (joint_value.expand(batch_size, joint_num, joint_num) + torch.eye(joint_num,
                                                                                          device=self.device).unsqueeze(
            0).expand(batch_size, joint_num, joint_num) * delta).reshape(batch_size, -1, joint_num)
        joint_value = torch.cat([joint_value, d_joint_value], dim=1).reshape(batch_size * (joint_num + 1), joint_num)

        if base_trans is not None:
            base_trans = base_trans.unsqueeze(1).expand(batch_size, (joint_num + 1), 4, 4).reshape(
                batch_size * (joint_num + 1), 4, 4)
        sdf, _ = self.get_whole_body_sdf_batch(points, joint_value, model, base_trans=base_trans, use_derivative=False,
                                               used_links=used_links)
        sdf = sdf.reshape(batch_size, (joint_num + 1), -1)
        d_sdf = (sdf[:, 1:, :] - sdf[:, :1, :]) / delta
        return sdf[:, 0, :], d_sdf.transpose(1, 2)

    def get_whole_body_normal_with_joints_grad_batch(self, points, joint_value, model, base_trans=None,
                                                     used_links=None):
        """
        Get the normal vector of the whole body with respect to the joints

        :param points: (batch_size, 3)
        :param joint_value: (batch_size, joint_num)
        :param model: the trained RDF model
        :param base_trans: the transformation matrix of base pose, (1, 4, 4)
        :param used_links: the links to be used, list of link names
        :return:
        """
        delta = 0.001
        batch_size = joint_value.shape[0]
        joint_num = joint_value.shape[1]
        link_num = len(self.robot.get_link_list())
        joint_value = joint_value.unsqueeze(1)

        d_joint_value = (joint_value.expand(batch_size, joint_num, joint_num) +
                         torch.eye(joint_num, device=self.device).unsqueeze(0).expand(batch_size, joint_num,
                                                                                      joint_num) * delta).reshape(
            batch_size, -1, joint_num)
        joint_value = torch.cat([joint_value, d_joint_value], dim=1).reshape(batch_size * (joint_num + 1), joint_num)

        if base_trans is not None:
            base_trans = base_trans.unsqueeze(1).expand(batch_size, (joint_num + 1), 4, 4).reshape(
                batch_size * (joint_num + 1), 4, 4)
        sdf, normal = self.get_whole_body_sdf_batch(points, joint_value, model, base_trans=base_trans,
                                                    use_derivative=True, used_links=used_links)
        normal = normal.reshape(batch_size, (joint_num + 1), -1, 3).transpose(1, 2)
        return normal  # normal size: (batch_size,N,8,3) normal[:,:,0,:] origin normal vector normal[:,:,1:,:] derivatives with respect to joints

    def visualize_reconstructed_whole_body(self, model, trans_list, tag):
        """
        Visualize the reconstructed whole body

        :param model: the trained RDF model
        :param trans_list: the transformation matrices of all links
        :param tag: the tag of the mesh, e.g., 'BP_8'
        :return:
        """
        view_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        scene = trimesh.Scene()

        for link_name, mesh_dict in self.link_mesh_map.items():
            for _, origin_mf in mesh_dict.items():
                if origin_mf is not None:
                    mesh_name = origin_mf["params"]["name"]
                    mf = os.path.join(self.wdf_dir, f"output_meshes/{tag}_{mesh_name}.stl")
                    mesh = trimesh.load(mf)
                    mesh_dict = model[mesh_name]
                    offset = mesh_dict['offset'].cpu().numpy()
                    scale = mesh_dict['scale']
                    mesh.vertices = mesh.vertices * scale + offset

                    all_related_link = [key for key in trans_list.keys() if link_name in key]
                    try:
                        related_link = all_related_link[-1]
                    except:
                        pass
                    mesh.apply_transform(trans_list[related_link].squeeze().cpu().numpy())
                    mesh.apply_transform(view_mat)
                    scene.add_geometry(mesh)
        scene.show()


def job(task):
    link_name, mesh_name, save_path, geom_info = task
    # 调用采样函数
    data = sample_sdf_points(link_name, mesh_name, save_path, geom_info)
    # 返回数据，包括 link_name 和 geom_name 以便后续保存
    return data


def sample_sdf_points(link_name, mesh_name, save_path, geom_info):
    """
    Sample SDF points for both simple shapes and mesh geometries.
    This function handles simple shapes like sphere, box, cylinder, and capsule,
    and complex shapes like meshes using the mesh_to_sdf library.

    :param link_name: The name of the link.
    :param mesh_name: The name of the mesh.
    :param save_path: Directory to save the sampled points and SDF values.
    :param geom_info: The geometry information dictionary.
    :return: The sampled SDF data.
    """
    geom_type = geom_info['type']
    geom_params = geom_info['params']

    print(f'Sampling points for mesh {mesh_name}_{geom_type}...')

    if geom_type == "sphere":
        # Sphere sampling
        radius = geom_params['radius']
        center = np.array(geom_params['position'])
        scale = radius * 2  # Set scale to be the radius of the sphere

        near_points = np.random.rand(500000, 3) * 2.0 - 1.0
        random_points = np.random.rand(500000, 3) * 2.0 - 1.0
        # Compute SDF using scaled parameters
        near_sdf = np.array(sdf_sphere(near_points, center - center, radius / scale))
        random_sdf = np.array(sdf_sphere(random_points, center - center, radius / scale))

    elif geom_type == "box":
        # Box sampling
        half_size = np.array(geom_params['extents']) / 2.0
        center = np.array(geom_params['position'])
        scale = np.linalg.norm(half_size * 2)  # Use the diagonal length as the scale for the box

        near_points = np.random.rand(500000, 3) * 2.0 - 1.0
        random_points = np.random.rand(500000, 3) * 2.0 - 1.0
        # Compute SDF using scaled parameters
        near_sdf = np.array(sdf_box(near_points, center - center, half_size / scale))
        random_sdf = np.array(sdf_box(random_points, center - center, half_size / scale))

    elif geom_type == "cylinder":
        # Cylinder sampling
        radius = geom_params['radius']
        height = geom_params['height']
        center = np.array(geom_params['position'])
        scale = max(radius, height)  # Use the larger of radius or half-height as the scale

        near_points = np.random.rand(500000, 3) * 2.0 - 1.0
        random_points = np.random.rand(500000, 3) * 2.0 - 1.0
        # Compute SDF using scaled parameters
        near_sdf = np.array(sdf_cylinder(near_points, center - center, radius / scale, height / scale))
        random_sdf = np.array(sdf_cylinder(random_points, center - center, radius / scale, height / scale))

    elif geom_type == "capsule":
        # Capsule sampling
        radius = geom_params['radius']
        from_point = np.array(geom_params['from'])
        to_point = np.array(geom_params['to'])
        height = geom_params['height']
        center = (from_point + to_point) / 2.0  # Use midpoint as approximate center for capsules
        scale = max(radius, height)  # Use the larger of radius or half-height as the scale

        near_points = np.random.rand(500000, 3) * 2.0 - 1.0
        random_points = np.random.rand(500000, 3) * 2.0 - 1.0
        # Compute SDF using scaled parameters
        near_sdf = np.array(
            sdf_capsule(near_points, (from_point - center) / scale, (to_point - center) / scale, radius / scale,
                        height / scale))
        random_sdf = np.array(
            sdf_capsule(random_points, (from_point - center) / scale, (to_point - center) / scale, radius / scale,
                        height / scale))

    elif geom_type == "mesh":
        # Complex mesh sampling using mesh_to_sdf
        mesh = trimesh.load(geom_params.get('mesh_path'))
        if isinstance(mesh, trimesh.Scene):
            combined_mesh = mesh.geometry.values()
            mesh = trimesh.util.concatenate(combined_mesh)
        center = mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(mesh.vertices - center, axis=1))
        mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)

        # Sample points near the surface (as in DeepSDF)
        near_points, near_sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,
                                                                    number_of_points=500000,
                                                                    surface_point_method='scan',
                                                                    sign_method='normal',
                                                                    scan_count=100,
                                                                    scan_resolution=400,
                                                                    sample_point_count=10000000,
                                                                    normal_sample_count=100,
                                                                    min_size=0.0,
                                                                    return_gradients=False)

        # Sample points randomly within the bounding box [-1,1]
        random_points = np.random.rand(500000, 3) * 2.0 - 1.0
        random_sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                             random_points,
                                             surface_point_method='scan',
                                             sign_method='normal',
                                             bounding_radius=None,
                                             scan_count=100,
                                             scan_resolution=400,
                                             sample_point_count=10000000,
                                             normal_sample_count=100)
    else:
        raise ValueError(f"Unsupported geometry type {geom_type}.")

    # Save the sampled data
    data = {
        'link_name': link_name,
        'mesh_name': mesh_name,
        'near_points': near_points,
        'near_sdf': near_sdf,
        'random_points': random_points,
        'random_sdf': random_sdf,
        'center': center,
        'scale': scale
    }
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f'voxel_128_{link_name}_{mesh_name}_{geom_info.get("type")}.npy'), data)

    print(f'Sampling points for mesh {mesh_name} finished!')
    return data
