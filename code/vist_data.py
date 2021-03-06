"""Datalodader classes for the VIST dataset.
"""

import os
import functools
import re
import pdb
import random
import pickle

import tqdm
import numpy as np
import torch
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms

import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from miscc.utils import valid_img_path, valid_np_img

ENCODINGS_FILE = 'test_text_encodings.pt'
PICKLE_FILE = 'test_annotations.pickle'
LABEL_FILE = 'test_labels.npy'


class StoryDataset(torch.utils.data.Dataset):
    """Dataloader class for stories.

    Loads a batch of images and encoded sentences of a story and concatenate them.

    Note:
        This class takes `desc_dir` as a parameter to align with original dataset modules
        created by the StoryGAN author.

    Attributes:
        img_dir (str): Directory containing dataset images
        encodings (file): File containing encoded sentences in an 1-D array
        stories (file): File containing tuples of (image_id, encoding_id)
                       in an 1-D array
        labels (file): File containing one-hot K-vector of the most common nouns
                       in the dataset sentences
        transforms (str): Pytorch transform defined in `main.py`
        video_len (int): Number of frames in a story
    """

    def __init__(self, img_dir, desc_dir, transform, video_len=4, is_train=True):
        self.img_dir = img_dir
        self.encodings = torch.load(os.path.join(desc_dir, ENCODINGS_FILE))
        self.stories = pickle.load(open(os.path.join(desc_dir, PICKLE_FILE), "rb"))
        self.labels = np.load(os.path.join(desc_dir, LABEL_FILE))
        self.transforms = transform
        self.video_len = video_len

    def __getitem__(self, item):
        image = []
        des = []
        labels = []
        story = self.stories[item]

        # Append images, descriptions, and labels
        for i in range(self.video_len):
            # images
            image_id, enc_idx = story[i]
            img_path = valid_img_path('%s/%s.jpg' % (self.img_dir, image_id))
            im = Image.open(img_path)
            image.append(valid_np_img(im, image_id))
            im.close()            

            # descriptions
            desc = self.encodings[enc_idx]
            des.append(desc.unsqueeze(0))

            # labels
            labels.append(self.labels[enc_idx])


        # image is T x H x W x C
        # After transform, image is C x T x H x W    
        image_numpy = image
        image = self.transforms(image_numpy)
        des = np.concatenate(des, axis = 0) 
        des = torch.tensor(des)
        label = np.array(labels)

        return {'images': image, 'description': des, 'label': label}

    def __len__(self):
        return len(self.stories)


class ImageDataset(torch.utils.data.Dataset):
    """Dataloader class for an image.

    Loads a single image-sentence pair in a story.

    Note:
        This class takes `desc_dir` as a parameter to align with original dataset modules
        created by the StoryGAN author.

    Attributes:
        img_dir (str): Directory containing dataset images
        encodings (file): File containing encoded sentences in an 1-D array
        stories (file): File containing tuples of (image_id, encoding_id)
                       in an 1-D array
        labels (file): File containing one-hot K-vector of the most common nouns
                       in the dataset sentences                       
        transforms (str): Pytorch transform defined in `main.py`
        video_len (int): Number of frames in a story
    """

    def __init__(self, img_dir, desc_dir, transform, video_len=4, is_train=True):
        self.img_dir = img_dir
        self.encodings = torch.load(os.path.join(desc_dir, ENCODINGS_FILE))
        self.stories = pickle.load(open(os.path.join(desc_dir, PICKLE_FILE), "rb"))
        self.labels = np.load(os.path.join(desc_dir, LABEL_FILE))        
        self.transforms = transform
        self.video_len = video_len

    def __getitem__(self, item):
        """Returns:
            {
            images (3-D numpy array): an image of a scene (storylet)
            description (1-D Tensor): an encoded sentence of a scene (storylet)
            label (1-D numpy array): a lable of a scene (storylet)
            content (2-D Tensor): concatenated encoded sentences of a whole story
                                  (self.video_len x encoding size)
            }
        """
        # description
        story = self.stories[item]
        image_id, enc_idx = random.choice(story)
        desc = self.encodings[enc_idx]
        des = desc

        # image
        path = valid_img_path('%s/%s.jpg' % (self.img_dir, image_id))
        im = Image.open(path)
        image = valid_np_img(im, image_id)
        image = self.transforms(image)
        im.close()           

        # content
        content = []
        for i in range(self.video_len):
            image_id, enc_idx = story[i]
            desc = self.encodings[enc_idx]          
            content.append(desc.unsqueeze(0))
        content = np.concatenate(content, 0)      
        content = torch.tensor(content)         

        content = torch.zeros_like(content) # TODO: Nullify story context        

        # label
        label = self.labels[enc_idx]

        return {'images': image, 'description': des, 'label': label, 'content': content}

    def __len__(self):
        return len(self.stories)