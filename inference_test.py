# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

# import dataset
import models
#####/media/zxl/E/zxl/deep-high-resolution-net.pytorch/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)
    #######3mpii
    #/media/zxl/E/zxl/deep-high-resolution-net.pytorch/experiments/mpii/hgcpef/hg8_256x256_d256x3_adam_lr2.5e-4.yaml
    ##
    #########coco
    #/media/zxl/E/zxl/deep-high-resolution-net.pytorch/experiments/coco/hgcpef/w32_256x192_adam_lr1e-3.yaml
    ##____/media/zxl/E/zxl/deep-high-resolution-net.pytorch/experiments/coco/hgcpef/w32_384x288_adam_lr1e-3.yaml
    ##___/media/zxl/E/zxl/deep-high-resolution-net.pytorch/experiments/mpii/hgcpef/hg8_256x256_d256x3_adam_lr2.5e-4.yaml
    #/media/zxl/E/zxl/deep-high-resolution-net.pytorch/experiments/coco/hgcpef/w32_384x288_adam_lr1e-3.yaml
    #/media/zxl/E/zxl/deep-high-resolution-net_paper1.pytorch/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
    #/media/zxl/E/zxl/deep-high-resolution-net_paper1.pytorch/experiments/mpii/hgcpef/hg8_256x256_d256x3_adam_lr2.5e-4.yaml
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='/media/zxl/E/zxl/deep-high-resolution-net_paper1.pytorch/experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

import numpy as np

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # model = EfficientNet.from_pretrained(‘efficientnet - b0’)
    # device = torch.device(cuda)
    # model.to(device)
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print('mean_syn',mean_syn)


if __name__ == '__main__':
    main()
