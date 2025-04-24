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

import torch
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import GradL1Loss, SILogLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np

# taken from evaluation 


import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


"""
Licence MIT

Copyright (c) 2022 Intelligent Systems Lab Org

The following classes and methods have been adapted from ZoeDepth to our application. It has originally been written by author Shariq Farooq Bhat.

The original code: https://github.com/isl-org/ZoeDepth/tree/main
"""
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}



def compute_scale_and_shift(prediction, target, mask):
    """
    Compute scale and shift to align the 'prediction' to the 'target' using the 'mask'.

    This function solves the system Ax = b to find the scale (x_0) and shift (x_1) that aligns the prediction to the target. 
    The system matrix A and the right hand side b are computed from the prediction, target, and mask.

    Args:
        prediction (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        mask (torch.Tensor): Mask that indicates the zones to evaluate. 

    Returns:
        tuple: Tuple containing the following:
            x_0 (torch.Tensor): Scale factor to align the prediction to the target.
            x_1 (torch.Tensor): Shift to align the prediction to the target.
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (0, 1))
    a_01 = torch.sum(mask * prediction, (0, 1))
    a_11 = torch.sum(mask, (0, 1))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (0, 1))
    b_1 = torch.sum(mask * target, (0, 1))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0
    
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def compute_errors(gt, pred):
    """
    Compute the 5 error metrics between the ground truth and the prediction:
    - Absolute relative error (abs_rel)
    - Squared relative error (sq_rel)
    - Root mean squared error (rmse)
    - Root mean squared error on the log scale (rmse_log)
    - Scale invariant log error (silog)

    Args:
        gt (numpy.ndarray): Ground truth values.
        pred (numpy.ndarray): Predicted values.

    Returns:
        dict: Dictionary containing the following metrics:
            'abs_rel': Absolute relative error
            'sq_rel': Squared relative error
            'rmse': Root mean squared error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """


    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    return dict(abs_rel=abs_rel, rmse=rmse, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_metrics(gt, pred, mask_score, sport):
    """
    This code creates a mask of the same size as the ground truth and prediction, 
    and then modifies the mask based on the sport and the values in the ground truth and prediction. 
    The mask is used to exclude certain areas from the evaluation analysis.

    Args:
        pred (torch.Tensor): Predicted values.
        gt (torch.Tensor): Ground truth values.
        sport (str): The sport to evaluate the predictions on. Can be "basket" or "foot".
        mask_score (bool): Whether to mask the score area in football images.

    Returns:
        dict: Dictionary containing the error metrics computed by the 'compute_errors' function, 
        applied to the areas of the ground truth and prediction indicated by the mask.
    """
    mask = np.ones((1080, 1920), dtype=np.bool_)

    if sport == "basket":
        print("here")
        mask[870:1016, 1570:1829] = False
    if sport == "foot" and mask_score:
        print("in the problem")
        mask[70:122, 95:612] = False


    pred = pred.squeeze().cpu().numpy()
    mask[pred <= 0] = False
    mask[np.isinf(pred)] = False
    mask[np.isnan(pred)] = False

    gt = gt.squeeze().cpu().numpy()
    mask[gt <= 0] = False
    mask[np.isinf(gt)] = False
    mask[np.isnan(gt)] = False


    return compute_errors(gt[mask], pred[mask])






class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """
        # print(batch)
        images, depths_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        # dataset = batch['dataset'][0]
        # print("image shape: {}".format(images.shape))
        b, c, h, w = images.size()
        # NOTE: bxy: set the mask according to the min_depth and max_depth, 
        # eseecpcially the min_depth mask make 
        # import pdb; pdb.set_trace()
        dataset = 'sn'
        min_depth=DATASETS_CONFIG[dataset]['min_depth']
        max_depth=DATASETS_CONFIG[dataset]['max_depth']
        mask = torch.logical_and(depths_gt > min_depth, depths_gt < max_depth)
        # print('mask valida elements has: {}, the total elements are: {}'.format(mask.sum(), mask.shape[-1] * mask.shape[-2]))
        # mask = torch.ones(depths_gt.shape, dtype=depths_gt.dtype).to(self.device).to(torch.bool)
        losses = {}

        with amp.autocast(enabled=self.config.use_amp):

            output = self.model(images)
            pred_depths = output['metric_depth']

            l_si, pred = self.silog_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            loss = self.config.w_si * l_si
            losses[self.silog_loss.name] = l_si

            if self.config.w_grad > 0:
                l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad
            else:
                l_grad = torch.Tensor([0])

        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        # if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
        #     # -99 is treated as invalid depth in the log_images function and is colored grey.
        #     depths_gt[torch.logical_not(mask)] = -99

        #     self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train",
        #                     min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        #     if self.config.get("log_rel", False):
        #         self.log_images(
        #             scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            pred_depths = m(x)['metric_depth']
        return pred_depths

    @torch.no_grad()
    def crop_aware_infer(self, x):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths


    #original
    # def validate_on_batch_(self, batch, val_step):
    #     images = batch['image'].to(self.device)
    #     depths_gt = batch['depth'].to(self.device)
    #     # dataset = batch['dataset'][0]
    #     dataset = 'sn'
    #     # mask = batch["mask"].to(self.device)
    #     mask = torch.ones(depths_gt.shape, dtype=depths_gt.dtype).to(self.device)
    #     if 'has_valid_depth' in batch:
    #         if not batch['has_valid_depth']:
    #             return None, None

    #     depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
    #     mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
    #     if dataset == 'nyu':
    #         pred_depths = self.crop_aware_infer(images)
    #     else:
    #         pred_depths = self.eval_infer(images)
    #     pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)

    #     with amp.autocast(enabled=self.config.use_amp):
    #         l_depth = self.silog_loss(
    #             pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)

    #     metrics = compute_metrics(depths_gt, pred_depths, **self.config)
    #     losses = {f"{self.silog_loss.name}": l_depth.item()}

    #     if val_step == 1 and self.should_log:
    #         depths_gt[torch.logical_not(mask)] = -99
            # self.log_images(rgb={"Input": images[0]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
            #                 min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses
    import torch.nn.functional as F
    def sn_test(self, preds, gts, sn_metric):
        # import pdb; pdb.set_trace()
        batch_size = preds.size(0)
        for i in range(batch_size):
            pred = preds[i:i+1, ...].squeeze()
            gt = gts[i:i+1, ...].squeeze() / 256.
            pred = nn.functional.interpolate(pred.unsqueeze(0).unsqueeze(0), (1080, 1920), mode='bilinear', align_corners=True).squeeze()
            mask = torch.ones(pred.shape, device=pred.device, dtype=pred.dtype)
            scale, shift = compute_scale_and_shift(pred, gt, mask)
            scaled_prediction = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
            scaled_prediction = scaled_prediction.squeeze()
            # gt = nn.functional.interpolate(gt.unsqueeze(0).unsqueeze(0), (1080, 1920), mode='bilinear', align_corners=True).squeeze()
            # scaled_prediction = nn.functional.interpolate(scaled_prediction.unsqueeze(0).unsqueeze(0), (1080, 1920), mode='bilinear', align_corners=True).squeeze()
            sn_metric.update(compute_metrics(gt, scaled_prediction, mask_score=False, sport='foot'))
        return sn_metric



    # TODO adap this to write a code impl off-line eval on single gpu without ddp
    def eval_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        self.model.eval()
        sn_metric = RunningAverageDict()
        with torch.no_grad():
            pred_depths = self.eval_infer(images)
            pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
            sn_metric = self.sn_test(pred_depths, depths_gt, sn_metric)
            metrics = sn_metric.get_value()
            
        return metrics

    def validate(self):
        assert self.config.distributed
        self.model.eval()
        num_samples = 0
        self.sn_metric = RunningAverageDict()       
        with torch.no_grad():
            for batch in tqdm(self.test_loader, leave=False, disable=(not self.should_log)):
                images = batch['image'].to(self.device)
                depths_gt = batch['depth'].to(self.device)
                pred_depths = self.eval_infer(images)
                # print('image shape: {}, pred_depths shape: {}'.format(images.shape, pred_depths.shape))
                self.sn_metric = self.sn_test(pred_depths, depths_gt, self.sn_metric)
                batch_size = images.shape[0]
                num_samples += batch_size

        # 汇总所有进程的统计
        self.sn_metric_stats = self.sn_metric.get_value()
        self.sn_metric_stats = {k: v * num_samples  for k, v in self.sn_metric_stats.items()} 
        for key in self.sn_metric_stats:
            tensor = torch.tensor(self.sn_metric_stats[key], device=torch.device(f'cuda:{self.config.rank}'))
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            self.sn_metric_stats[key] = tensor.item()
            
        # 汇总样本数
        num_samples_tensor = torch.tensor(num_samples, device=torch.device(f'cuda:{self.config.rank}'))
        torch.distributed.all_reduce(num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        num_samples = num_samples_tensor.item()

        # 计算全局平均
        self.sn_metric_stats = {k: v / num_samples  for k, v in self.sn_metric_stats.items()} 
        # self.rmse = self.sn_metric_stats['rmse']
        return self.sn_metric_stats
