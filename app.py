# From: https://huggingface.co/spaces/Sense-X/uniformer_video_demo/blob/main/app.py

import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from uniformer import uniformer_small
from kinetics_class_index import kinetics_classnames
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

import gradio as gr
from huggingface_hub import hf_hub_download


def get_index(num_frames, num_segments=16, dense_sample_rate=8):
    sample_range = num_segments * dense_sample_rate
    sample_pos = max(1, 1 + num_frames - sample_range)
    t_stride = dense_sample_rate
    start_idx = 0 if sample_pos == 1 else sample_pos // 2
    #print(start_idx, sample_pos, num_segments)
    
    offsets = np.array([
        (idx * t_stride + start_idx) %
        num_frames for idx in range(num_segments)
    ])
    #print(start_idx, num_frames)
    #print('offsets1', offsets)
    """ offsets = np.linspace(
        start=start_idx,
        stop=num_frames-1,
        num=num_segments,
        dtype=np.int
    ) """
    #print('offsets', offsets)
    """ print('offsets2',offsets)
    sys.exit() """
    return offsets + 1


def load_video(video_path):
    #vr = VideoReader(video_path, ctx=cpu(0))
    vr = [np.asarray(Image.open(x)) for x in video_path]#glob.glob(video_path + '*')]
    num_frames = len(vr)
    #frame_indices = get_index(num_frames, 16, 16)
    frame_indices = range(len(video_path))

    # transform
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    transform = T.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])
    #try:
    images_group = list()
    #print('num indices', len(frame_indices), frame_indices)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index])#.asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    """ except IndexError:
        print(frame_indices)
        sys.exit() """
    # The model expects inputs of shape: B x C x T x H x W
    #print('torch_imgs', torch_imgs.shape)
    TC, H, W = torch_imgs.shape
    torch_imgs = torch_imgs.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4)

    return torch_imgs
    

def inference(video):
    vid = load_video(video)
    
    prediction = model(vid)
    prediction = F.softmax(prediction, dim=1).flatten()

    return {kinetics_id_to_classname[str(i)]: float(prediction[i]) for i in range(400)}
    
