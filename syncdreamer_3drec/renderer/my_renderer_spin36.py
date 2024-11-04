import abc
import os
from pathlib import Path
from skimage.transform import rescale, resize
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from skimage.io import imread, imsave
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

from base_utils import read_pickle, concat_images_list
from renderer.neus_networks import SDFNetwork, RenderingNetwork, SingleVarianceNetwork, SDFHashGridNetwork, RenderingFFNetwork
from renderer.ngp_renderer import NGPNetwork
from util import instantiate_from_config

DEFAULT_RADIUS = np.sqrt(3)/2
DEFAULT_SIDE_LENGTH = 0.6
# 函数从由输入权重定义的概率密度函数（PDF）中采样点。
# 它使用逆变换采样从PDF的累积分布函数（CDF）生成样本。
# 根据`det`参数生成确定性或随机样本。
def sample_pdf(bins, weights, n_samples, det=True):
    device = bins.device
    dtype = bins.dtype
    # 这个实现来自NeRF
    # 获取pdf
    weights = weights + 1e-5  # 防止出现nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # 取均匀样本
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, dtype=dtype, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], dtype=dtype, device=device)

    # 反转CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

# 这个函数计算射线与球体的交点，返回最近和最远的交点距离。
# 它假设球体的半径为`radius`，默认值为`DEFAULT_RADIUS`。
def near_far_from_sphere(rays_o, rays_d, radius=DEFAULT_RADIUS):
    a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
    b = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = -b / a
    near = mid - radius
    far = mid + radius
    return near, far

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image

class BaseRenderer(nn.Module):
    def __init__(self, train_batch_num, test_batch_num):
        super().__init__()
        self.train_batch_num = train_batch_num
        self.test_batch_num = test_batch_num

    # 抽象方法，子类需要实现具体的渲染逻辑
    @abc.abstractmethod
    def render_impl(self, ray_batch, is_train, step):
        pass

    # 抽象方法，子类需要实现具体的渲染损失计算逻辑
    @abc.abstractmethod
    def render_with_loss(self, ray_batch, is_train, step):
        pass

    # 渲染函数，根据是否训练选择不同的batch大小，逐批处理射线数据
    def render(self, ray_batch, is_train, step):
        batch_num = self.train_batch_num if is_train else self.test_batch_num
        ray_num = ray_batch['rays_o'].shape[0]
        outputs = {}
        for ri in range(0, ray_num, batch_num):
            cur_ray_batch = {}
            for k, v in ray_batch.items():
                cur_ray_batch[k] = v[ri:ri + batch_num]
            cur_outputs = self.render_impl(cur_ray_batch, is_train, step)
            for k, v in cur_outputs.items():
                if k not in outputs: outputs[k] = []
                outputs[k].append(v)

        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0)
        return outputs

class NeuSRenderer(BaseRenderer):
    def __init__(self, train_batch_num, test_batch_num, lambda_eikonal_loss=0.1, use_mask=True,
                 lambda_rgb_loss=1.0, lambda_mask_loss=0.0, rgb_loss='soft_l1', coarse_sn=64, fine_sn=64):
        super().__init__(train_batch_num, test_batch_num)
        self.n_samples = coarse_sn
        self.n_importance = fine_sn
        self.up_sample_steps = 4
        self.anneal_end = 200
        self.use_mask = use_mask
        self.lambda_eikonal_loss = lambda_eikonal_loss
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.rgb_loss = rgb_loss

        self.sdf_network = SDFNetwork(d_out=257, d_in=3, d_hidden=256, n_layers=8, skip_in=[4], multires=6, bias=0.5, scale=1.0, geometric_init=True, weight_norm=True)
        self.color_network = RenderingNetwork(d_feature=256, d_in=9, d_out=3, d_hidden=256, n_layers=4, weight_norm=True, multires_view=4, squeeze_out=True)
        self.default_dtype = torch.float32
        self.deviation_network = SingleVarianceNetwork(0.3)

    @torch.no_grad()
    def get_vertex_colors(self, vertices):
        """
        @param vertices:  n,3
        @return:
        """
        V = vertices.shape[0]
        bn = 20480
        verts_colors = []
        material_seg = []
        with torch.no_grad():
            for vi in range(0, V, bn):
                verts = torch.from_numpy(vertices[vi:vi+bn].astype(np.float32)).cuda()
                feats = self.sdf_network(verts)[..., 1:]
                gradients = self.sdf_network.gradient(verts)  # ...,3
                gradients = F.normalize(gradients, dim=-1)
                colors_and_seg = self.color_network(verts, gradients, gradients, feats)
                colors = colors_and_seg['rgb']
                colors = torch.clamp(colors,min=0,max=1).cpu().numpy()
                verts_colors.append(colors)
                material=colors_and_seg['seg']
                material_seg.append(material.cpu().numpy())
        verts_colors = (np.concatenate(verts_colors, 0)*255).astype(np.uint8)
        material_seg = np.concatenate(material_seg, 0)
        return verts_colors, material_seg

    # 给定固定的 inv_s 的情况下进行上采样
    def upsample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        inner_mask = self.get_inner_mask(pts)
        # radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = inner_mask[:, :-1] | inner_mask[:, 1:]
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], dtype=self.default_dtype, device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], dtype=self.default_dtype, device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    # 将新的 z 值与原有的 z 值和 sdf 进行拼接
    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            device = pts.device
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1).to(device)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    # 采样深度
    def sample_depth(self, rays_o, rays_d, near, far, perturb):
        n_samples = self.n_samples
        n_importance = self.n_importance
        up_sample_steps = self.up_sample_steps
        device = rays_o.device

        # sample points
        batch_size = len(rays_o)
        z_vals = torch.linspace(0.0, 1.0, n_samples, dtype=self.default_dtype, device=device)   # sn
        z_vals = near + (far - near) * z_vals[None, :]            # rn,sn

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]).to(device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / n_samples

        # Up sample
        with torch.no_grad():
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            sdf = self.sdf_network.sdf(pts).reshape(batch_size, n_samples)

            for i in range(up_sample_steps):
                rn, sn = z_vals.shape
                inv_s = torch.ones(rn, sn - 1, dtype=self.default_dtype, device=device) * 64 * 2 ** i
                new_z_vals = self.upsample(rays_o, rays_d, z_vals, sdf, n_importance // up_sample_steps, inv_s)
                z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == up_sample_steps))

        return z_vals

    #  计算 sdf 和 alpha
    def compute_sdf_alpha(self, points, dists, dirs, cos_anneal_ratio, step):
        # points [...,3] dists [...] dirs[...,3]
        sdf_nn_output = self.sdf_network(points)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.sdf_network.gradient(points)  # ...,3
        inv_s = self.deviation_network(points).clip(1e-6, 1e6)  # ...,1
        inv_s = inv_s[..., 0]

        true_cos = (dirs * gradients).sum(-1)  # [...]
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # [...]
        return alpha, gradients, feature_vector, inv_s, sdf

    #  获取退火值
    def get_anneal_val(self, step):
        if self.anneal_end < 0:
            return 1.0
        else:
            return np.min([1.0, step / self.anneal_end])

    #  获取内部掩码
    def get_inner_mask(self, points):
        return torch.sum(torch.abs(points)<=DEFAULT_SIDE_LENGTH,-1)==3

    #  渲染逻辑
    def render_impl(self, ray_batch, is_train, step):
        near, far = near_far_from_sphere(ray_batch['rays_o'], ray_batch['rays_d'])
        rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
        z_vals = self.sample_depth(rays_o, rays_d, near, far, is_train)

        batch_size, n_samples = z_vals.shape

        # section length in original space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # rn,sn-1
        dists = torch.cat([dists, dists[..., -1:]], -1)  # rn,sn
        mid_z_vals = z_vals + dists * 0.5

        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_z_vals.unsqueeze(-1) # rn, sn, 3
        inner_mask = self.get_inner_mask(points)

        dirs = rays_d.unsqueeze(-2).expand(batch_size, n_samples, 3)
        dirs = F.normalize(dirs, dim=-1)
        device = rays_o.device
        has_seg = True

        num_classes = 3 #material number
        # num_classes = int(np.max(mask_seg)) + 1

        alpha, sampled_color, gradient_error, normal = torch.zeros(batch_size, n_samples, dtype=self.default_dtype, device=device), \
            torch.zeros(batch_size, n_samples, 3, dtype=self.default_dtype, device=device), \
            torch.zeros([batch_size, n_samples], dtype=self.default_dtype, device=device), \
            torch.zeros([batch_size, n_samples, 3], dtype=self.default_dtype, device=device)
        sample_seg= torch.zeros([batch_size, n_samples, num_classes], dtype=self.default_dtype, device=device)
        weights = torch.zeros(batch_size, n_samples, dtype=self.default_dtype, device=device)
        # weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], dtype=self.default_dtype, device=device), 1. - alpha + 1e-7], -1), -1)[..., :-1]

        if torch.sum(inner_mask) > 0:
            cos_anneal_ratio = self.get_anneal_val(step) if is_train else 1.0
            alpha[inner_mask], gradients, feature_vector, inv_s, sdf = self.compute_sdf_alpha(points[inner_mask], dists[inner_mask], dirs[inner_mask], cos_anneal_ratio, step)
            if has_seg:
                out_dict = self.color_network(points[inner_mask], gradients, -dirs[inner_mask], feature_vector)
                sampled_color[inner_mask] = out_dict['rgb']
                sample_seg[inner_mask] = out_dict['seg']
                weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], dtype=self.default_dtype, device=device), 1. - alpha + 1e-7], -1), -1)[..., :-1]
                seg = (sample_seg * weights[..., None]).sum(dim=1)
                # seg = out_dict['seg']
            else:
                sampled_color[inner_mask] = self.color_network(points[inner_mask], gradients, -dirs[inner_mask], feature_vector)
            # Eikonal loss
            gradient_error[inner_mask] = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2 # rn,sn
            normal[inner_mask] = F.normalize(gradients, dim=-1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], dtype=self.default_dtype, device=device), 1. - alpha + 1e-7], -1), -1)[..., :-1]  # rn,sn
        mask = torch.sum(weights,dim=1).unsqueeze(-1) # rn,1
        color = (sampled_color * weights[..., None]).sum(dim=1) + (1 - mask) # add white background
        normal = (normal * weights[..., None]).sum(dim=1)

        outputs = {
            'rgb': color,  # rn,3
            'gradient_error': gradient_error,  # rn,sn
            'inner_mask': inner_mask,  # rn,sn
            'normal': normal,  # rn,3
            'mask': mask,  # rn,1
            'seg': seg if has_seg else None,
        }
        return outputs

    # 渲染损失计算逻辑
    def render_with_loss(self, ray_batch, is_train, step):
        render_outputs = self.render(ray_batch, is_train, step)

        has_seg = True
        rgb_gt = ray_batch['rgb']
        rgb_pr = render_outputs['rgb']
        if has_seg:
            rgb_seg_gt = ray_batch['rgb']
            rgb_gt, seg_gt = rgb_seg_gt[:, :3], rgb_seg_gt[:, -1]
            seg_pr = render_outputs['seg']
            seg_gt = seg_gt.long()
            mask_one_hot = F.one_hot(seg_gt, num_classes=3).float()
            # breakpoint()
            seg_loss = F.cross_entropy(seg_pr, mask_one_hot, reduction='none')
            # breakpoint()
            seg_loss = torch.mean(seg_loss)
            
        
        if self.rgb_loss == 'soft_l1':
            epsilon = 0.001
            rgb_loss = torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        elif self.rgb_loss =='mse':
            rgb_loss = F.mse_loss(rgb_pr, rgb_gt, reduction='none')
        else:
            raise NotImplementedError
        rgb_loss = torch.mean(rgb_loss)

        eikonal_loss = torch.sum(render_outputs['gradient_error'] * render_outputs['inner_mask']) / torch.sum(render_outputs['inner_mask'] + 1e-5)
        loss = rgb_loss * self.lambda_rgb_loss + eikonal_loss * self.lambda_eikonal_loss
        loss_batch = {
            'eikonal': eikonal_loss,
            'rendering': rgb_loss,
            # 'mask': mask_loss,
        }
        if self.lambda_mask_loss>0 and self.use_mask:
            mask_loss = F.mse_loss(render_outputs['mask'], ray_batch['mask'], reduction='none').mean()
            loss += mask_loss * self.lambda_mask_loss
            loss_batch['mask'] = mask_loss
        return loss, loss_batch


class RendererTrainer(pl.LightningModule):
    def __init__(self, image_path, total_steps, warm_up_steps, log_dir, elevation, distance, train_batch_fg_num=0,
                 use_cube_feats=False, cube_ckpt=None, cube_cfg=None, cube_bound=0.5,
                 train_batch_num=4096, test_batch_num=8192, use_warm_up=True, use_mask=True,
                 lambda_rgb_loss=1.0, lambda_mask_loss=0.0, renderer='neus',
                 # used in neus
                 lambda_eikonal_loss=0.1,
                 coarse_sn=64, fine_sn=64):
        super().__init__()
        self.num_images = 36
        self.image_size = 256
        self.log_dir = Path(log_dir)/'images'
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.train_batch_num = train_batch_num
        self.train_batch_fg_num = train_batch_fg_num
        self.test_batch_num = test_batch_num
        self.image_path = image_path
        self.total_steps = total_steps
        self.warm_up_steps = warm_up_steps
        self.use_mask = use_mask
        self.lambda_eikonal_loss = lambda_eikonal_loss
        self.lambda_rgb_loss = lambda_rgb_loss
        self.lambda_mask_loss = lambda_mask_loss
        self.use_warm_up = use_warm_up
        self.elevation = elevation
        self.distance = distance
        self.use_cube_feats, self.cube_cfg, self.cube_ckpt = use_cube_feats, cube_cfg, cube_ckpt

        self._init_dataset()
        if renderer=='neus':
            self.renderer = NeuSRenderer(train_batch_num, test_batch_num,
                                         lambda_rgb_loss=lambda_rgb_loss,
                                         lambda_eikonal_loss=lambda_eikonal_loss,
                                         lambda_mask_loss=lambda_mask_loss,
                                         coarse_sn=coarse_sn, fine_sn=fine_sn)
        else:
            raise NotImplementedError
        self.validation_index = 0

    # 构建射线批次
    def _construct_ray_batch(self, images_info):
        image_num = images_info['images'].shape[0]
        color_ch = images_info['images'].shape[-1]
        _, h, w, _ = images_info['images'].shape
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.float()[None, :, :, :].repeat(image_num, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(image_num, h * w, 2)
        coords = torch.cat([coords, torch.ones(image_num, h * w, 1, dtype=torch.float32)], 2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        rays_d = coords @ torch.inverse(images_info['Ks']).permute(0, 2, 1)
        poses = images_info['poses']  # imn,3,4
        R, t = poses[:, :, :3], poses[:, :, 3:]
        rays_d = rays_d @ R
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = -R.permute(0,2,1) @ t # imn,3,3 @ imn,3,1
        rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn,h*w,3
        # breakpoint()
        ray_batch = {
            'rgb': images_info['images'].reshape(image_num*h*w,color_ch),
            'mask': images_info['masks'].reshape(image_num*h*w,1),
            'rays_o': rays_o.reshape(image_num*h*w,3).float(),
            'rays_d': rays_d.reshape(image_num*h*w,3).float(),
        }
        return ray_batch

    # 加载模型
    @staticmethod
    def load_model(cfg, ckpt):
        config = OmegaConf.load(cfg)
        model = instantiate_from_config(config.model)
        print(f'loading model from {ckpt} ...')
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda().eval()
        return model

    #  初始化数据集
    def _init_dataset(self):
        mask_predictor = BackgroundRemoval()
        self.K, _, _, _, _ = read_pickle(f'meta_info/camera-16.pkl')

        self.images_info = {'images': [] ,'masks': [], 'Ks': [], 'poses':[]}
        seg_maskdir = '/dtu/blackhole/11/180913/seg_material/merged_results'

        for index in range(self.num_images):
            rgb = imread(f'{self.image_path}/{index}.png')
            # mask_seg = imread(os.path.join(seg_maskdir,'chair%i'%(index+1),'clean_masks/0','1_mask.png'))
            mask_seg = imread(os.path.join(seg_maskdir, 'chair%i' % (index + 1) + '.png'))
            mask_seg = cv2.resize(mask_seg, (256, 256), interpolation=cv2.INTER_NEAREST)

            # predict mask
            if self.use_mask:
                imsave(f'{self.log_dir}/input-{index}.png', rgb)
                masked_image = mask_predictor(rgb)
                imsave(f'{self.log_dir}/masked-{index}.png', masked_image)
                mask = masked_image[:,:,3].astype(np.float32)/255
            else:
                h, w, _ = rgb.shape
                mask = np.zeros([h,w], np.float32)

            # rgb = rgb.astype(np.float32)
            num_classes = int(np.max(mask_seg)) + 1

            rgb_seg = np.concatenate([rgb, mask_seg[...,None]], -1)
            K = np.copy(self.K)

            e, a = self.elevation, index*np.pi*2/self.num_images
            row1 = np.asarray([np.sin(e)*np.cos(a),np.sin(e)*np.sin(a),-np.cos(e)])
            row0 = np.asarray([-np.sin(a),np.cos(a), 0])
            row2 = np.cross(row0, row1)
            R = np.stack([row0,row1,row2],0)
            t = np.asarray([0, 0, self.distance])
            pose = np.concatenate([R,t[:,None]],1)
            add_seg = True
            if add_seg:
                rgb = rgb_seg

            self.images_info['images'].append(torch.from_numpy(rgb.astype(np.float32))) # h,w,3
            self.images_info['masks'].append(torch.from_numpy(mask.astype(np.float32))) # h,w
            self.images_info['Ks'].append(torch.from_numpy(K.astype(np.float32)))
            self.images_info['poses'].append(torch.from_numpy(pose.astype(np.float32)))

        # 数据堆叠
        for k, v in self.images_info.items(): self.images_info[k] = torch.stack(v, 0) # stack all values

        self.train_batch = self._construct_ray_batch(self.images_info)
        self.train_batch_pseudo_fg = {}
        pseudo_fg_mask = torch.sum(self.train_batch['rgb']>0.99,1)!=3
        for k, v in self.train_batch.items():
            self.train_batch_pseudo_fg[k] = v[pseudo_fg_mask]
        self.train_ray_fg_num = int(torch.sum(pseudo_fg_mask).cpu().numpy())
        self.train_ray_num = self.num_images * self.image_size ** 2
        self._shuffle_train_batch()
        self._shuffle_train_fg_batch()

    #  打乱训练批次
    def _shuffle_train_batch(self):
        self.train_batch_i = 0
        shuffle_idxs = torch.randperm(self.train_ray_num, device='cpu') # shuffle
        for k, v in self.train_batch.items():
            self.train_batch[k] = v[shuffle_idxs]

    #  打乱前景训练批次
    def _shuffle_train_fg_batch(self):
        self.train_batch_fg_i = 0
        shuffle_idxs = torch.randperm(self.train_ray_fg_num, device='cpu') # shuffle
        for k, v in self.train_batch_pseudo_fg.items():
            self.train_batch_pseudo_fg[k] = v[shuffle_idxs]

    #  训练步骤
    def training_step(self, batch, batch_idx):
        train_ray_batch = {k: v[self.train_batch_i:self.train_batch_i + self.train_batch_num].cuda() for k, v in self.train_batch.items()}
        self.train_batch_i += self.train_batch_num
        if self.train_batch_i + self.train_batch_num >= self.train_ray_num: self._shuffle_train_batch()

        if self.train_batch_fg_num>0:
            train_ray_batch_fg = {k: v[self.train_batch_fg_i:self.train_batch_fg_i+self.train_batch_fg_num].cuda() for k, v in self.train_batch_pseudo_fg.items()}
            self.train_batch_fg_i += self.train_batch_fg_num
            if self.train_batch_fg_i + self.train_batch_fg_num >= self.train_ray_fg_num: self._shuffle_train_fg_batch()
            for k, v in train_ray_batch_fg.items():
                train_ray_batch[k] = torch.cat([train_ray_batch[k], v], 0)

        loss, loss_batch = self.renderer.render_with_loss(train_ray_batch, is_train=True, step=self.global_step)
        self.log_dict(loss_batch, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log('step', self.global_step, prog_bar=True, on_step=True, on_epoch=False, logger=False, rank_zero_only=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss

    #  切片图像信息
    def _slice_images_info(self, index):
        return {k:v[index:index+1] for k, v in self.images_info.items()}

    #  验证步骤
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.global_rank==0:
                # we output an rendering image
                images_info = self._slice_images_info(self.validation_index)
                self.validation_index += 1
                self.validation_index %= self.num_images

                test_ray_batch = self._construct_ray_batch(images_info)
                test_ray_batch = {k: v.cuda() for k,v in test_ray_batch.items()}
                test_ray_batch['near'], test_ray_batch['far'] = near_far_from_sphere(test_ray_batch['rays_o'], test_ray_batch['rays_d'])
                render_outputs = self.renderer.render(test_ray_batch, False, self.global_step)

                process = lambda x: (x.cpu().numpy() * 255).astype(np.uint8)
                h, w = self.image_size, self.image_size
                rgb = torch.clamp(render_outputs['rgb'].reshape(h, w, 3), max=1.0, min=0.0)
                mask = torch.clamp(render_outputs['mask'].reshape(h, w, 1), max=1.0, min=0.0)
                mask_ = torch.repeat_interleave(mask, 3, dim=-1)
                output_image = concat_images_list(process(rgb), process(mask_))
                if 'normal' in render_outputs:
                    normal = torch.clamp((render_outputs['normal'].reshape(h, w, 3) + 1) / 2, max=1.0, min=0.0)
                    normal = normal * mask # we only show foregound normal
                    output_image = concat_images_list(output_image, process(normal))

                # save images
                imsave(f'{self.log_dir}/images/{self.global_step}.jpg', output_image)

    #  配置优化器
    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW([{"params": self.renderer.parameters(), "lr": lr},], lr=lr)

        def schedule_fn(step):
            total_step = self.total_steps
            warm_up_step = self.warm_up_steps
            warm_up_init = 0.02
            warm_up_end = 1.0
            final_lr = 0.02
            interval = 1000
            times = total_step // interval
            total_step = 100
            times = 1
            ratio = np.power(final_lr, 1/times)
            if step<warm_up_step:
                learning_rate = (step / warm_up_step) * (warm_up_end - warm_up_init) + warm_up_init
            else:
                learning_rate = ratio ** (step // interval) * warm_up_end
            return learning_rate

        if self.use_warm_up:
            scheduler = [{
                    'scheduler': LambdaLR(opt, lr_lambda=schedule_fn),
                    'interval': 'step',
                    'frequency': 1
                }]
        else:
            scheduler = []
        return [opt], scheduler
