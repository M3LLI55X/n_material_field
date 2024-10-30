import argparse

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import trimesh
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from skimage.io import imsave
from tqdm import tqdm

import mcubes

from base_utils import read_pickle, output_points
from renderer.my_renderer_spin36 import NeuSRenderer, DEFAULT_SIDE_LENGTH
from util import instantiate_from_config

class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups

def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0,color_func=None):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    
    u_all = []
    seg_nums = 6
    seg_index = 2
    for seg_index in range(seg_nums):
        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                        val = query_func(pts).detach()
                        vertex_colors, vertex_seg = color_func(pts.cpu().numpy())
                        vertex_seg_results = np.argmax(vertex_seg, axis=-1)
                        mask = vertex_seg_results == seg_index
                        val[mask] = 0
                        outside_mask = torch.norm(pts,dim=-1)>=1.0
                        val[outside_mask]=outside_val
                        val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        u_all.append(u)
    return u_all


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, color_func, outside_val=1.0):
    u_all = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val,color_func=color_func)
    # geometries = []
    # n = len(u_all)
    
    # for u in u_all:

    vertices, triangles = mcubes.marching_cubes(u_all[0], threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    vertex_colors, vertex_seg = color_func(vertices)
    # vertex_seg_results = np.argmax(vertex_seg, axis=-1)  # [n,]
    # n_class = np.unique(vertex_seg_results)
    # mask = vertex_seg_results == n_class
    # vertices_cp = vertices.copy()
    # vertex_colors_cp = vertex_colors.copy()
    # vertices_cp[~mask] = 0
    # vertex_colors_cp[~mask] = 0
    # geometries.append((vertices, triangles, vertex_colors))
    return vertices, triangles, vertex_colors

def extract_mesh(model, output, resolution=512):
    if not isinstance(model.renderer, NeuSRenderer): return
    bbox_min = -torch.ones(3) * DEFAULT_SIDE_LENGTH
    bbox_max = torch.ones(3) * DEFAULT_SIDE_LENGTH
    with torch.no_grad():
        vertices, triangles, vertex_colors = extract_geometry(bbox_min, bbox_max, resolution, 0, lambda x: model.renderer.sdf_network.sdf(x), lambda x: model.renderer.get_vertex_colors(x))

    # for i, (vertices_cp, triangles_cp, vertex_colors_cp, n_class) in enumerate(geometries):
    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors)
    # mesh.export(f'{output}/class_{n_class}_mesh_{i}.ply')
    mesh.export(f'{output}/class_mesh.ply')


# def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, color_func, outside_val=1.0):
#     u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
#     vertices, triangles = mcubes.marching_cubes(u, threshold)
#     b_max_np = bound_max.detach().cpu().numpy()
#     b_min_np = bound_min.detach().cpu().numpy()

#     vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
#     vertex_colors,vertex_seg = color_func(vertices)
#     vertex_seg_results = np.argmax(vertex_seg, axis=-1) # [n,]

#     # 将顶点坐标从体数据的网格索引空间转换到实际的三维空间坐标。
#     geometries = []
#     for n_class in np.unique(vertex_seg_results):
#         mask = vertex_seg_results == n_class
#         vertices_cp = vertices.copy()
#         vertex_colors_cp = vertex_colors.copy()
#         vertices_cp[~mask] = 0
#         vertex_colors_cp[~mask] = 0
#         geometries.append((vertices_cp, triangles, vertex_colors_cp, n_class))
#     return geometries


# def extract_mesh(model, output, resolution=512):
#     if not isinstance(model.renderer, NeuSRenderer): return
#     bbox_min = -torch.ones(3) * DEFAULT_SIDE_LENGTH
#     bbox_max = torch.ones(3) * DEFAULT_SIDE_LENGTH
#     with torch.no_grad():
#         # sdf_field = extract_fields(bbox_min, bbox_max, resolution, lambda x: model.renderer.sdf_network.sdf(x))
#         geometries = extract_geometry(bbox_min, bbox_max, resolution, 0, lambda x: model.renderer.sdf_network.sdf(x), lambda x: model.renderer.get_vertex_colors(x))

#     # output geometry
#     for vertices_cp, triangles, vertex_colors_cp, n_class in geometries:
#         mesh = trimesh.Trimesh(vertices_cp, triangles, vertex_colors=vertex_colors_cp)
#         mesh.export(str(f'{output}/{n_class}_mesh.ply'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-b', '--base', type=str, default='my_neus_spin36.yaml')
    parser.add_argument('-l', '--log', type=str, default='output')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('-e', '--elevation', type=float, default=np.pi/6)
    parser.add_argument('-d', '--distance', type=float, default=1.5)
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

    # configs
    cfg = OmegaConf.load(opt.base)
    name = opt.name
    log_dir, ckpt_dir = Path(opt.log) / name, Path(opt.log) / name / 'ckpt'
    print(str(log_dir))
    cfg.model.params['image_path'] = opt.image_path
    cfg.model.params['log_dir'] = str(log_dir)

    # setup
    log_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    trainer_config = cfg.trainer
    callback_config = cfg.callbacks
    model_config = cfg.model
    data_config = cfg.data

    data_config.params.seed = opt.seed
    data = instantiate_from_config(data_config)
    data.prepare_data()
    data.setup('fit')


    model_config.params.elevation = opt.elevation
    model_config.params.distance = opt.distance
    model = instantiate_from_config(model_config,)
    model.cpu()
    model.learning_rate = model_config.base_lr

    # logger
    logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard_logs')
    callbacks=[]
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:06}", verbose=True, save_last=True, every_n_train_steps=callback_config.save_interval))

    # trainer
    trainer_config.update({
        "accelerator": "gpu", "check_val_every_n_epoch": 500000,
        "benchmark": True, "num_sanity_val_steps": 0,
        "devices": 1, "gpus": opt.gpus, 
    })
    if opt.fp16:
        trainer_config['precision']=16

    if opt.resume:
        callbacks.append(ResumeCallBacks())
        trainer_config['resume_from_checkpoint'] = str(ckpt_dir / 'last.ckpt')
    else:
        if (ckpt_dir / 'last.ckpt').exists():
            raise RuntimeError(f"checkpoint {ckpt_dir / 'last.ckpt'} existing ...")
    trainer = Trainer.from_argparse_args(args=argparse.Namespace(), **trainer_config, logger=logger, callbacks=callbacks)

    trainer.fit(model, data)

    model = model.cuda().eval()

    # render_images(model, log_dir, opt.elevation)
    extract_mesh(model, log_dir)

if __name__=="__main__":
    main()