# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)

import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

import numpy as np
from PIL import Image



import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import Normalize
from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.deepth_2025 import SNDataset
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log



@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images) #, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]) )#, **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2.unsqueeze(0), [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    # print("Saved raw depth to", fpath)


@torch.no_grad()
def infer_samples(model, test_loader, config):
    # import pdb; pdb.set_trace()
    model.eval()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        # image, depth = sample['image'], sample['depth']
        image = sample['image']

        image = image.cuda()
        # import pdb; pdb.set_trace()
        image =  F.interpolate(image, size=(392, 714), mode='bilinear') #image.unsqueeze(1); .squeeze(1)
        # import pdb; pdb.set_trace()
        out_path = sample['out_path'][0]
        # print(out_path)
        # image, depth = image.cuda(), depth.cuda()
        # depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset='sn', focal=focal)
        # save the pred
        # colorize the pred
        
        
        pred_resized = F.interpolate(pred, size=(1080, 1920), mode='bilinear', align_corners=True)
        print('pred_resized min: {}, max: {}'.format(pred_resized.min(), pred_resized.max()))
        pred_resized = torch.clamp(pred_resized, 0, 65535/256)
        save_raw_16bit(pred_resized, out_path)
        

def main(args, config):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': 300})
    
    
    model.load_state_dict({k: v for k, v in torch.load("/home/lsk/Depth-Anything-V2/metric_depth/exp/SNDataset/latest.pth", map_location='cpu').items() if 'pretrained' in k}, strict=True)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    test_loader = DepthDataLoader(config, 'infer', ).data
    model = model.cuda()
    infer_samples(model, test_loader, config)
    # metrics = evaluate(model, test_loader, config)
    # print(f"{colors.fg.green}")
    # print(metrics)
    # print(f"{colors.reset}")
    # metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    # return metrics


def eval_model(model_name, pretrained_resource, output_dir, split, dataset='sn', **kwargs):

    # # Load default pretrained resource defined in config if not set
    # overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    overwrite = {**kwargs}
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    print(config)
    print(f"Evaluating {model_name} on {dataset}...")

    # model = build_model(config)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs["vitl"], 'max_depth': 300})
    
    
    model.load_state_dict({k: v for k, v in torch.load("/home/lsk/Depth-Anything-V2/metric_depth/exp/SNDataset/latest.pth", map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    test_loader = DepthDataLoader(config, split, output_dir=output_dir).data
    model = model.cuda()
    infer_samples(model, test_loader, config)



    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    parser.add_argument("-s", "--split", type=str,
                    required=False, default="test_infer", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    
    parser.add_argument("-o", "--output_dir", type=str,
                    required=False, default="infer", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    # import pdb; pdb.set_trace()
    eval_model(args.model, args.pretrained_resource, args.output_dir, args.split,  **overwrite_kwargs)
