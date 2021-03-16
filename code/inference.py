from __future__ import print_function

import argparse
import os
import random
import sys
import pprint
import functools
import importlib

import dateutil
import dateutil.tz
import numpy as np
import pdb

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import PIL

from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p, video_transform
from evaluater import GANEvaluator

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


if __name__ == "__main__":
    # Configure main arguments
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='optional config file',
                        default='./cfg/clevr.yml',
                        type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default='')
    parser.add_argument('--desc_path', dest='desc_path', type=str, default='')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='./output')
    args = parser.parse_args()   
         
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    data = importlib.import_module('{}_data'.format(cfg.DATASET_NAME))        

    # Configure CUDA
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    num_gpu = len(cfg.GPU_ID.split(','))      

    # Set output path
    output_dir = os.path.join(args.output_dir,
                    '%s_%s/' % (cfg.DATASET_NAME, cfg.CONFIG_NAME))
    test_sample_save_dir = output_dir + 'test/'
    if not os.path.exists(test_sample_save_dir):
        os.makedirs(test_sample_save_dir)

    # Define image transformation
    n_channels = 3      
    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize( (cfg.IMSIZE, cfg.IMSIZE) ),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    # DataLoader
    testset = data.StoryDataset(args.img_dir,
                                args.desc_path,
                                video_transforms,
                                cfg.VIDEO_LEN,
                                False)    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

    # Run inference
    algo = GANEvaluator(output_dir, test_sample_save_dir)
    algo.evaluate(testloader)