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

# import pdb;pdb.set_trace()
parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim')
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

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



def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (392, 714)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        trainset = VKITTI2('dataset/splits/vkitti2/train.txt', 'train', size=size)
    elif args.dataset == 'SNDataset':
        from torch.utils.data import ConcatDataset
        train_1 = SNDataset(depth_transform = Normalize([0.0], [256]), split='train')
        train_2 = SNDataset(depth_transform = Normalize([0.0], [256]), split='val')
        train_3 = SNDataset(depth_transform = Normalize([0.0], [256]), split='test_train')
        #train_4 = SNDataset(depth_transform = Normalize([0.0], [256]), split='Train',data_root='/home/lsk/Depth-Anything-V2/SoccerNet/depth-basketball')
        training_samples = ConcatDataset([train_1, train_2, train_3])
        print('train samples: {}'.format(len(training_samples)))
        # Dataset(
        #     config, mode, transform=transform, device=device)

        trainsampler = torch.utils.data.distributed.DistributedSampler(
                training_samples)

        trainloader = DataLoader(training_samples,
                            batch_size=args.bs,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            #    prefetch_factor=2,
                            sampler=trainsampler)

    # trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    elif args.dataset == 'SNDataset':
        valset = SNDataset(split='test_test',depth_transform = Normalize([0.0],[256]))
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset,
                                   batch_size=args.bs,
                                   shuffle=(valsampler is None),
                                   num_workers=4,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=valsampler)
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    if args.pretrained_from:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    criterion = SiLogLoss().cuda(local_rank)
    
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    best_rmse = 100000000

    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()
            
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
            
            pred = model(img)
            
            loss = criterion(pred, depth)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        model.eval()
        
        # results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
        #            'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
        #            'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
        nsamples = torch.tensor([0.0]).cuda()
        sn_metric_stats = {} #还没填
        num_samples = 0
        sn_metric = RunningAverageDict()

        import torch.nn.functional as F
        def sn_test(preds, gts, sn_metric):
            # import pdb; pdb.set_trace()
            batch_size = preds.size(0)
            for i in range(batch_size):
                pred = preds[i:i+1, ...].squeeze()
                gt = gts[i:i+1, ...].squeeze() / 256.
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), (1080, 1920), mode='bilinear', align_corners=True).squeeze()
                mask = torch.ones(pred.shape, device=pred.device, dtype=pred.dtype)
                scale, shift = compute_scale_and_shift(pred, gt, mask)
                scaled_prediction = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
                scaled_prediction = scaled_prediction.squeeze()
                # gt = nn.functional.interpolate(gt.unsqueeze(0).unsqueeze(0), (1080, 1920), mode='bilinear', align_corners=True).squeeze()
                # scaled_prediction = nn.functional.interpolate(scaled_prediction.unsqueeze(0).unsqueeze(0), (1080, 1920), mode='bilinear', align_corners=True).squeeze()
                sn_metric.update(compute_metrics(gt, scaled_prediction, mask_score=False, sport='foot'))
            return sn_metric
        
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                
                images, depths_gt, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
                
                # with torch.no_grad():
                pred_depths = model(images)
                pred_depths = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
                # valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
                # pred = pred.unsqueeze(0)
                # if valid_mask.sum() < 10:
                #     continue
                
                # cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
                
                # for k in results.keys():
                #     results[k] += cur_results[k]
                nsamples += 1

                sn_metric = sn_test(pred_depths, depths_gt, sn_metric)
                print(sn_metric)
                #这对吗
                for k in sn_metric.keys():
                    sn_metric_stats[k] += sn_metric[k]
                metrics = sn_metric.get_value()

                import pdb; pdb.set_trace()
                batch_size = images.shape[0]
                num_samples += batch_size
            
        torch.distributed.barrier()
        
        # for k in results.keys():
        #     dist.reduce(results[k], dst=0)
        # dist.reduce(nsamples, dst=0)
        for k in sn_metric_stats.keys():
            dist.reduce(sn_metric_stats[k], dst=0)
        dist.reduce(nsamples, dst=0)
        
        # # 汇总所有进程的统计
        # sn_metric_stats = sn_metric.get_value()
        # sn_metric_stats = {k: v * num_samples  for k, v in sn_metric_stats.items()} 
        # for key in sn_metric_stats:
        #     tensor = torch.tensor(sn_metric_stats[key], device=torch.device(f'cuda:{config.rank}'))
        #     torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        #     sn_metric_stats[key] = tensor.item()
            
        # # 汇总样本数
        # num_samples_tensor = torch.tensor(num_samples, device=torch.device(f'cuda:{self.config.rank}'))
        # torch.distributed.all_reduce(num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        # num_samples = num_samples_tensor.item()

        # # 计算全局平均
        # metrics = {k: v / num_samples  for k, v in sn_metric_stats.items()} 

        
        # print(f"Step {self.step} - Test Losses: " + ", ".join([f"Test/{name}: {tloss}" for name, tloss in test_losses.items()]))
        print(f"Step {i} - Metrics: " + ", ".join([f"Metrics/{k}: {v}" for k, v in sn_metric_stats.items()]))

        rmse = metrics['rmse']
        def save_checkpoint(self, filename):
            if not self.should_write:
                return
            root = args.save_path #self.config.save_dir
            if not os.path.isdir(root):
                os.makedirs(root)

            fpath = os.path.join(root, filename)
            m = self.model.module if self.config.multigpu else self.model
            
            torch.save(
                {
                    "model": m.state_dict(),
                    "optimizer": None,  # TODO : Change to self.optimizer.state_dict() if resume support is needed, currently None to reduce file size
                    "epoch": epoch
                }, fpath)
            # torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        
        
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(sn_metric_stats.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in sn_metric_stats.values()])))
            logger.info('==========================================================================================')
            print()
            
            for name, metric in sn_metric_stats.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        # for k in results.keys():
        #     if k in ['d1', 'd2', 'd3']:
        #         previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        #     else:
        #         previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            if (rmse < best_rmse): #and args.should_write:
                save_checkpoint(f"{args.experiment_id}_{rmse}.pth")
            best_rmse = rmse
            # checkpoint = {
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'epoch': epoch,
            #     # 'previous_best': previous_best,
            # }
            # torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        
        model.train()


if __name__ == '__main__':
    main()