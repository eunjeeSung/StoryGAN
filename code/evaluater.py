from __future__ import print_function

import os
import time
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from six.moves import range

from miscc.config import cfg
from miscc.utils import save_story_results, weights_init


class GANEvaluator:
    def __init__(self, output_dir, test_dir = None):
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.video_len = cfg.VIDEO_LEN
        self.output_dir = os.path.join(output_dir, 'Test')
        self.test_dir = test_dir
        self.netG = self.load_networks()

    def load_networks(self):
        from model import StoryGAN, D_IMG, D_STY
        netG = StoryGAN(self.video_len)
        try:
            snapshot = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(snapshot['state_dict'])
            epoch, iteration = snapshot['epoch'], snapshot['iteration']
            print('Load from: ', cfg.NET_G)
        except Exception as e:
            print('{}: Net_G Not Found'.format(e))
            raise
    
        if cfg.CUDA:
            netG.cuda()
        return netG

    def batch_inference(self, storyloader):
        for i, data in enumerate(tqdm(storyloader), 0):
            # Get story results
            st_real_cpu, st_motion_input, st_content_input = self._get_st_inputs(data)
            lr_fake, st_fake = self._sample_stories(st_motion_input, st_content_input)

            # Save results
            self._save_story_results(st_real_cpu, lr_fake, st_fake, i, self.output_dir)
        print("DONE: Batch inference")
        return st_fake

    def single_inference(self, st_motion_input):
        """Takes `st_motion_input` of size `(sentence_num, dim)`
        """
        # TODO: Take category labels as input
        st_content_input_shape = [1, self.video_len, st_motion_input.shape[-1]]
        st_content_input = torch.zeros(st_content_input_shape)
        st_motion_input = st_motion_input.unsqueeze(0)
        if cfg.CUDA:
            st_motion_input = st_motion_input.cuda()
            st_content_input = st_content_input.cuda()       

        lr_fake, st_fake = self._sample_stories(st_motion_input, st_content_input)
        return st_fake

    def _get_st_inputs(self, data):
        st_batch = data

        st_real_cpu = st_batch['images']
        st_motion_input = st_batch['description']
        st_content_input = st_batch['description']
        st_catelabel = st_batch['label']
        st_real_imgs = Variable(st_real_cpu)
        st_motion_input = Variable(st_motion_input)
        st_content_input = Variable(st_content_input)

        if cfg.CUDA:
            st_real_imgs = st_real_imgs.cuda()
            st_motion_input = st_motion_input.cuda()
            st_content_input = st_content_input.cuda()
            st_catelabel = st_catelabel.cuda()                
        return st_real_cpu, st_motion_input, st_content_input

    def _sample_stories(self, st_motion_input, st_content_input):
        lr_fake, st_fake, _, _, _, _ = self.netG.sample_videos(st_motion_input, st_content_input)
        return lr_fake, st_fake

    def _save_story_results(self, st_real_cpu, lr_fake, st_fake, num, output_dir):
        save_story_results(st_real_cpu, st_fake, num, output_dir, test=True)
        if lr_fake is not None:
            save_story_results(None, lr_fake, num, output_dir, test=True)