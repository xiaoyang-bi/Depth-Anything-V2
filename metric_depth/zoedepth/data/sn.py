# Original Code from: https://github.com/prs-eth/graph-super-resolution
# by bxy
from pathlib import Path
import json
from PIL import Image
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import matplotlib
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .utils import  random_horizontal_flip, random_rotate # downsample, bicubic_with_mask, random_crop, random_rotate, random_resized_crop
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



def pad_to_nearest_64(x):
    """
    将输入张量的高和宽 pad 到最近的 64 的倍数，pad 值为 nan。
    支持 (C, H, W) 或 (N, C, H, W) 形状的张量。
    """
    # 判断输入是 (N, C, H, W) 还是 (C, H, W)
    if x.dim() == 4:
        N, C, H, W = x.shape
        batch_mode = True
    elif x.dim() == 3:
        C, H, W = x.shape
        batch_mode = False
    else:
        raise ValueError("只支持 (C, H, W) 或 (N, C, H, W) 形状的张量")
    # 计算 pad 后的目标高和宽
    H_new = ((H + 63) // 64) * 64
    W_new = ((W + 63) // 64) * 64

    pad_h = H_new - H
    pad_w = W_new - W

    # pad 格式: (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)

    # 只在最后两个维度 pad
    if batch_mode:
        # 先展平 batch 和 channel 维度，pad 后再 reshape 回去
        x_reshape = x.view(-1, H, W)
        x_padded = F.pad(x_reshape, pad, mode='constant', value=float('nan'))
        x_padded = x_padded.view(N, C, H_new, W_new)
    else:
        x_padded = F.pad(x, pad, mode='constant', value=float('nan'))

    return x_padded

class SNDataset(Dataset):

    def __init__(
            self,
            data_root: str = "/home/lsk/sn-depth-main/depth-2025",
            camera_dir = 'color',
            depth_gt_dir = 'depth_r',
            do_horizontal_flip=True,
            max_rotation_angle: int = 10,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform= Normalize([0.0], [256]),
            in_memory=True,
            split='train',
            crop_valid=False,
            output_dir='infer',
            **kwargs
    ):
        # self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.crop_valid = crop_valid

        self.camera_dir = camera_dir
        self.depth_gt_dir = depth_gt_dir
        self.split = split
        self.output_dir = output_dir
        # import pdb; pdb.set_trace()
        self.data_dir = Path(data_root) /  split

        if self.split != 'challenge':
            self.samples = self._collect_samples(self.data_dir)
        else:
             self.samples = self._collect_samples_color(self.data_dir)
        
        



    def _collect_samples(self, data_dir):
        """收集所有匹配的color/depth图像对"""
        samples = []
        for game_dir in data_dir.glob("game_*"):
            for video_dir in game_dir.glob("video_*"):
                color_dir = video_dir / self.camera_dir
                depth_gt_dir = video_dir / self.depth_gt_dir
                # depth_source_dir = video_dir / self.depth_source_dir
                if color_dir.exists() and depth_gt_dir.exists():
                    image_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
                    for img_file in color_dir.glob("*.*"):
                        if img_file.suffix.lower() in image_ext:
                            depth_gt_file = depth_gt_dir / img_file.name
                            if depth_gt_file.exists() :
                                samples.append((str(img_file), str(depth_gt_file))) # bxy
        return samples
    
    
    def _collect_samples_color(self, data_dir):
        """收集所有匹配的color/depth图像对"""
        samples = []
        for game_dir in data_dir.glob("game_*"):
            for video_dir in game_dir.glob("video_*"):
                color_dir = video_dir / self.camera_dir
                if color_dir.exists():
                    image_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
                    for img_file in color_dir.glob("*.*"):
                        if img_file.suffix.lower() in image_ext:
                            samples.append( str(img_file) ) 
        return samples

    def __getitem__(self, index):

        im_index = index
        if self.split != 'challenge':
            color_path, depth_path = self.samples[im_index]
        else:
            color_path = self.samples[im_index]
            
        image = torch.from_numpy(np.array(Image.open(color_path).convert('RGB')).astype('float32')).permute(2, 0, 1) / 255.
        image =  F.interpolate(image.unsqueeze(1), size=(384, 704), mode='bilinear').squeeze(1)

        
        if ('infer' not in self.split) and (self.split != 'challenge'):
            depth_pil = Image.open(depth_path)
            depth_np = np.array(depth_pil).astype('float32')
            if depth_np.ndim == 3:
                depth_np = depth_np[:, :, 0]
            depth_map = torch.from_numpy(depth_np).unsqueeze(0)
                
            outputs = [image, depth_map] 
            if self.split == 'train':
                if self.do_horizontal_flip:
                    outputs = random_horizontal_flip(outputs)
                # outputs = random_rotate(outputs, self.max_rotation_angle, self.rotation_interpolation,
                #                         crop_valid=False)
                # outputs[0] = self.color_aug(outputs[0])

            if self.image_transform is not None:
                outputs[0] = self.image_transform(outputs[0])
            
            if self.depth_transform is not None:
                outputs[1] = self.depth_transform(outputs[1])
                
            image, depth_map = outputs
            sample = {'image': image, 'depth': depth_map}  #'focal':xxx, 'mask': xxx}
        else:
            # import pdb; pdb.set_trace()
            output_dir = Path(color_path).parent.parent/ self.output_dir
            output_dir.mkdir(exist_ok=True)
            out_path = str(output_dir / color_path.split('/')[-1])
            sample = {'image': image, 'out_path': out_path}
        return sample

    
    def __len__(self):
        return len(self.samples)
        # if self.split == 'train':
        #     return len(self.samples)
        # else:
        #     return 50
        # return 100
def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        value = value * 0.


    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def get_sn_loader(config, batch_size=1, **kwargs):
    # print(config)
    # print(data_dir_root)
    dataset = SNDataset(depth_transform = Normalize([0.0], [256])) #fixed
    return DataLoader(dataset, batch_size, **kwargs)

def to_pil(img):
    # (3, H, W) -> (H, W, 3)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    # (H, W, 4) or (H, W, 3) 直接返回
    if img.ndim == 3 and img.shape[2] in [3, 4]:
        return Image.fromarray(img)
    # (1, H, W) -> (H, W)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    # (1, H, W, 4) -> (H, W, 4)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    # (3, H, W, 4) -> 只取第一个通道 (H, W, 4)
    if img.ndim == 4 and img.shape[0] == 3:
        img = img[0]
    return Image.fromarray(img)

if __name__ == '__main__':
    from torchvision.transforms import Normalize
    # import pdb; pdb.set_trace()
    dataset = SNDataset( depth_transform = Normalize([0.0], [256]), split='challenge')
    num_len = len(dataset)
    print("num len: {}".format(num_len))

    # 创建保存目录
    os.makedirs('visualization', exist_ok=True)
    # import pdb; pdb.set_trace()
    for i in range(min(num_len, 10)):  # 查看前10个样本
        data = dataset[i]
        print('img shape: {}'.format(data['image'].shape))
        # depth_map = data['depth']
        # print("depth min: {}, max: {}".format(depth_map.min(), depth_map.max()))
        # guide, gt = data['image'], data['depth']
        # source = guide
        # # guide, gt = data['guide'], data['y'], data['source']
        # print("Original shapes:", guide.shape, gt.shape)
        
        # # 将 torch.Tensor 转换为 numpy.ndarray
        # guide_np = guide.numpy() if torch.is_tensor(guide) else guide
        # gt_np = gt.numpy() if torch.is_tensor(gt) else gt
        # source_np = source.numpy() if torch.is_tensor(source) else source
        
        # # 处理 guide (假设是 (0,1) 范围的 float32)
        # if isinstance(guide_np, np.ndarray) and guide_np.dtype in [np.float32, np.float64]:
        #     guide_np = (guide_np * 255).astype(np.uint8)
        # elif isinstance(guide_np, np.ndarray) and guide_np.dtype == np.uint8:
        #     pass  # 已经是 0-255 范围
        # else:
        #     raise ValueError(f"Unexpected guide type: {type(guide_np)}, dtype: {guide_np.dtype}")
        
        # # 处理 gt 和 source (假设 colorize 返回的是 (0,255) 的 uint8)
        # gt_np = colorize(gt_np, cmap='magma_r')
        # source_np = colorize(source_np, cmap = 'magma_r')
        # print("After colorize shapes:", guide_np.shape, gt_np.shape, source_np.shape)
        # guide_pil = to_pil(guide_np)
        # gt_pil = to_pil(gt_np)
        # source_pil = to_pil(source_np)
        
        # # 保存图像
        # from pathlib import Path
        # # output_dir = 'visualization_zoedepth'
        # guide_pil.save(f'basketball/sample_{i}_guide.png')
        # gt_pil.save(f'basketball/sample_{i}_gt.png')
        # source_pil.save(f'basketball/sample_{i}_source.png')