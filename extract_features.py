"""
COMP5623M Coursework on Image Caption Generation


Forward pass through Flickr8k image data to extract and save features from
pretrained CNN.

"""


import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms

from models import EncoderCNN
from datasets import Flickr8k_Images
from utils import *
from config import *
import matplotlib.pyplot as plt
import torchvision
import datetime

lines = read_lines(TOKEN_FILE_TRAIN)
# see what is in lines
# print(lines[:2])

#########################################################################
#
#       QUESTION 1.1 Text preparation
# 
#########################################################################

image_ids, cleaned_captions = parse_lines(lines)
# to check the results after writing the cleaning function
print(image_ids[:20:5])
print(cleaned_captions[:20:5])

vocab = build_vocab(cleaned_captions)
# to check the results
print("Number of words in vocab:", vocab.idx)

# sample each image once
image_ids = image_ids[::5]

print(image_ids[0:5])

# crop size matches the input dimensions expected by the pre-trained ResNet
data_transform = transforms.Compose([ 
    transforms.Resize(224), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                         (0.229, 0.224, 0.225))])

dataset_train = Flickr8k_Images(
    image_ids=image_ids,
    transform=data_transform,
)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=False,
    num_workers=4,
)

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderCNN().to(device)
###############################################
#---------testcode-----below-----------
print("len(dataset_train)", len(dataset_train))
print(train_loader)
#just check the images


def timshow(x):
    xa = np.transpose(x.numpy(),(1,2,0))
    plt.imshow(xa)
    plt.show()
    return xa

# get some random training images

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

dataiter = iter(train_loader)
images = dataiter.next()


# show images and labels
timshow(torchvision.utils.make_grid(images))
print(images.shape)

#---------testcode-----above-----------
#############################################

#########################################################################
#
#        QUESTION 1.2 Extracting image features
# 
#########################################################################
features = []


# TODO loop through all image data, extracting features and saving them
# no gradients needed
dataiter_images = iter(train_loader)
images = dataiter.next()
print(type(images))
with torch.no_grad():
    for i,images in enumerate(train_loader):
        starttime_per_epoch = datetime.datetime.now()
        # print("images type:", type(images))
        # print(images.shape)
        images = images.to(device, non_blocking=True)
        features_in_batch = model(images)
        features_in_batch = torch.squeeze(features_in_batch)
        features_in_batch = features_in_batch.cpu()
        features_in_batch = features_in_batch.numpy().tolist()
        for feature in features_in_batch:
          features.append(feature)
        endtime_per_epoch = datetime.datetime.now()
        print(f"time consuming per epoch：{(endtime_per_epoch - starttime_per_epoch).total_seconds() : .3f} ")
# to check your results, features should be dimensions [len(train_set), 2048]
# convert features to a PyTorch Tensor before saving

features = torch.tensor(features)
features = torch.squeeze(features)
print(features.shape)
print(features[0:2])


# save features
torch.save(features, "features.pt")


