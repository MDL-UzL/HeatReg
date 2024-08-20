import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import corresponding_points_alignment, sample_points_from_meshes
from pytorch3d.transforms import random_rotation as random_rotation_
from pytorch3d.structures import Meshes
import trimesh
from pointops import *
import pyvista as pv

def homogenous(kpts):
    B, N, _ = kpts.shape
    device = kpts.device
    return torch.cat([kpts, torch.ones(B, N, 1, device=device)], dim=2) 

def transform_points(T, kpts):
    return (T @ homogenous(kpts).permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3].contiguous()

def estimate_transform(src, dst, estimate_scale=0):
    T_ = corresponding_points_alignment(src, dst, estimate_scale=estimate_scale)
    T = torch.eye(4).to(src.device).unsqueeze(0).repeat(src.shape[0], 1, 1)
    T[:, :3, :3] = T_.R.permute(0, 2, 1)
    T[:, :3, 3] = T_.T
    return T


def random_rotation(N):
    rotation = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    rotation[:, :3, :3] = torch.stack([random_rotation_() for _ in range(N)])
    return rotation

def is_right_handed(T):
    vec1 = T[1] - T[0]
    vec2 = T[2] - T[0]
    vec3 = torch.cross(vec1, vec2)

    dot_product = torch.dot(vec3, T[2])
    
    return dot_product > 0

def angle_between(v1, v2):
    v1 = v1 / v1.norm()
    v2 = v2 / v2.norm()
    return torch.arccos(torch.clip(torch.dot(v1, v2), -1.0, 1.0))

def generate_mesh(kpts_tf):
    device = kpts_tf.device
    N1, N2, D = kpts_tf.shape
    
    kpts_tf_ = kpts_tf.clone()
    threshs = []
    for i in range(N1):
        thresh = kpts_tf[i, kpts_tf_[i, :, 0].argmin(), 1]
        threshs.append(thresh)
        kpts_tf_[i, kpts_tf_[i, :, 1]<thresh, 0] -= 100000
    kpts_tf_ = torch.stack([kpts_tf_[i, kpts_tf_[i, :, 0].sort()[1], :] for i in range(N1)])
    for i in range(N1):
        kpts_tf_[i, kpts_tf_[i, :, 1]<threshs[i], 0] += 100000
    kpts_tf_sorted = torch.zeros_like(kpts_tf_)
    kpts_tf_sorted[:, 0] = kpts_tf_[:, -1]
    for i in range(N1):
        y = kpts_tf[i].clone()
        x = kpts_tf_sorted[i:i+1, 0]
        idx, dist = knn_query(1, y, torch.tensor([y.shape[0]]).to(y.device), x, torch.tensor([x.shape[0]]).to(x.device))
        y[idx[0]] = 100000
        for j in range(N2-1):
            x = kpts_tf_sorted[i:i+1, j]
            idx, dist = knn_query(1, y, torch.tensor([y.shape[0]]).to(y.device), x, torch.tensor([x.shape[0]]).to(x.device))
            kpts_tf_sorted[i, j+1] = y[idx[0]]
            y[idx[0]] = 100000
            
    ind = torch.arange(N1 * N2).view(-1, N2)
    faces = []
    for j in range(N1-1):
        x = kpts_tf_sorted[j]
        y = kpts_tf_sorted[j+1]
        idx, dist = knn_query(1, y, torch.tensor([y.shape[0]]).to(y.device), x, torch.tensor([x.shape[0]]).to(x.device))
        faces.append(torch.cat([torch.cat([ind[j:j+1, [0+i, 1+i]], ind[j+1:j+2, [idx[i], min(idx[i]+1, N2-1)]].flip(1)], dim=1) for i in range(N2-1)]))
    for j in range(N1-1):
        x = kpts_tf_sorted[j+1]
        y = kpts_tf_sorted[j]
        idx, dist = knn_query(1, y, torch.tensor([y.shape[0]]).to(y.device), x, torch.tensor([x.shape[0]]).to(x.device))
        faces.append(torch.cat([torch.cat([ind[j+1:j+2, [0+i, 1+i]], ind[j:j+1, [idx[i], min(idx[i]+1, N2-1)]].flip(1)], dim=1) for i in range(N2-1)]))
    faces = torch.cat(faces)
    faces = torch.cat([torch.ones(faces.shape[0], 1, dtype=torch.long) * 4, faces], dim=1)
    mesh = pv.PolyData(kpts_tf_sorted.view(-1, 3).cpu().numpy(), faces)
    mesh = mesh.triangulate()
    mesh_qual = mesh.compute_cell_quality('aspect_ratio')['CellQuality']
    mask = (mesh_qual < 30)[:, np.newaxis].repeat(4,1).flatten()
    mesh.faces = mesh.faces[mask]
    
    mesh_pt = Meshes([kpts_tf_sorted.view(-1, 3)], [torch.tensor(mesh.faces).view(-1, 4)[:, 1:].to(device)])
    return mesh_pt

def resample_sweep(mesh_pt, num_points=None):
    return sample_points_from_meshes(mesh_pt, num_points).view(1, -1, 3)

def generate_mesh_mask(kpts_tf, margin=0.5):
    N1, N2, D = kpts_tf.shape
    
    kpts_tf_ = kpts_tf.clone()
    threshs = []
    for i in range(N1):
        thresh = kpts_tf[i, kpts_tf_[i, :, 0].argmin(), 1]
        threshs.append(thresh)
        kpts_tf_[i, kpts_tf_[i, :, 1]<thresh, 0] -= 100000
    kpts_tf_ = torch.stack([kpts_tf_[i, kpts_tf_[i, :, 0].sort()[1], :] for i in range(N1)])
    for i in range(N1):
        kpts_tf_[i, kpts_tf_[i, :, 1]<threshs[i], 0] += 100000
    kpts_tf_sorted = torch.zeros_like(kpts_tf_)
    kpts_tf_sorted[:, 0] = kpts_tf_[:, -1]
    for i in range(N1):
        y = kpts_tf[i].clone()
        x = kpts_tf_sorted[i:i+1, 0]
        idx, dist = knn_query(1, y, torch.tensor([y.shape[0]]).to(y.device), x, torch.tensor([x.shape[0]]).to(x.device))
        y[idx[0]] = 100000
        for j in range(N2-1):
            x = kpts_tf_sorted[i:i+1, j]
            idx, dist = knn_query(1, y, torch.tensor([y.shape[0]]).to(y.device), x, torch.tensor([x.shape[0]]).to(x.device))
            kpts_tf_sorted[i, j+1] = y[idx[0]]
            y[idx[0]] = 100000
            
    kpts_tf_sorted[:, :, 0] += margin
    
    support1 = kpts_tf_sorted[:, 0].clone()
    support1[:, 1] = 30
    
    support2 = kpts_tf_sorted[:, -1].clone()
    support2[:, 1] = -30
    
    support3 = support1.clone()
    support3[:, 0] = -15
    
    support4 = support2.clone()
    support4[:, 0] = -15
    
    support5 = kpts_tf_sorted[:, 0].clone()
    support6 = kpts_tf_sorted[:, -1].clone()
    
    support = torch.cat([support1, support2, support3, support4, support5, support6], dim=0)
    
    faces = []
    for j in range(N1-1):
        faces.append(torch.tensor([j+4*N1, j+1+4*N1, j+1+N1+4*N1, j+N1+4*N1]).unsqueeze(0))
    for j in range(N1-1):
        faces.append(torch.tensor([j+2*N1, j+1+2*N1, j+1+N1+2*N1, j+N1+2*N1]).unsqueeze(0))
    for j in range(N1-1):
        faces.append(torch.tensor([j+1*N1, j+1+1*N1, j+1+N1+4*N1, j+N1+4*N1]).unsqueeze(0))
    for j in range(N1-1):
        faces.append(torch.tensor([j+0*N1, j+1+0*N1, j+1+N1+3*N1, j+N1+3*N1]).unsqueeze(0))
    for j in range(N1-1):
        faces.append(torch.tensor([j+0*N1, j+1+0*N1, j+1+N1+1*N1, j+N1+1*N1]).unsqueeze(0))
    for j in range(N1-1):
        faces.append(torch.tensor([j+1*N1, j+1+1*N1, j+1+N1+2*N1, j+N1+2*N1]).unsqueeze(0))
    faces = torch.cat(faces)
    faces = torch.cat([torch.ones(faces.shape[0], 1, dtype=torch.long) * 4, faces], dim=1)
    faces = faces.view(-1)
    
    faces = torch.cat([torch.tensor([6, 0*N1, 2*N1, 3*N1, 1*N1, 5*N1, 4*N1]), faces])
    faces = torch.cat([torch.tensor([6, 0*N1+N1-1, 2*N1+N1-1, 3*N1+N1-1, 1*N1+N1-1, 5*N1+N1-1, 4*N1+N1-1]), faces])
    
    mesh = pv.PolyData(support.view(-1, 3).cpu().numpy(), faces)
    mesh = mesh.triangulate()
    
    return trimesh.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:])

def get_kpts_support(kpts_radius_fixed_tf, kptss_radius_target_tf_resampled, num_points):
    device = kpts_radius_fixed_tf.device
    mesh_mask = generate_mesh_mask(kpts_radius_fixed_tf)

    mask = ~torch.tensor(mesh_mask.contains(kptss_radius_target_tf_resampled.view(-1, 3).cpu().numpy())).to(device)
    mask = mask & (kptss_radius_target_tf_resampled.view(-1, 3)[:, 2] > kpts_radius_fixed_tf[:, 2].min()).to(device)
    mask = mask & (kptss_radius_target_tf_resampled.view(-1, 3)[:, 2] < kpts_radius_fixed_tf[:, 2].max()).to(device)
    
    kpts_support = kptss_radius_target_tf_resampled.view(-1,3)[mask]
    
    ind = farthest_point_sampling(kpts_support, torch.tensor([kpts_support.shape[0]]).to(kpts_support.device), torch.tensor([num_points]).to(kpts_support.device))
    
    kpts_support = kpts_support[ind]
    
    return kpts_support.unsqueeze(0)