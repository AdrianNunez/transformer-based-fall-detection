import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

import yaml
import torch
import random
import numpy as np
from model import VideoTransformer
from dataset import FallDataset, URFallDataset, UPURFallDataset, load_labels
from kinetics_class_index import kinetics_classnames

def show_msg(msg):
    print('-'*20)
    print(msg)
    print('-'*20)

seed = 7
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

with open("config_eval2.yaml", "r") as stream:
    config = yaml.safe_load(stream)

train_labels, dev_labels, test_labels = load_labels(config)
train_dataset = UPURFallDataset(config, train_labels, 'train')
dev_dataset = UPURFallDataset(config, dev_labels, 'dev')
print('Train samples: {}'.format(len(train_dataset.labels)))
print('Dev samples: {}'.format(len(dev_dataset.labels)))

model = VideoTransformer(config)
model.train(config, train_dataset, dev_dataset)
del train_dataset, dev_dataset, train_labels, dev_labels
torch.save(model.state_dict(), config["model_save_path"])

test_dataset = FallDataset(config, test_labels, 'test')
ur_fall_test_dataset = URFallDataset(config, 'test')

print('Test samples UP-Fall: {}'.format(len(test_dataset)))
print('Test samples UR-Fall: {}'.format(len(ur_fall_test_dataset)))

if config["binarise"]:
    show_msg('UP Fall test set results')
    model.test(config, test_dataset)
    show_msg('UR Fall test set results')
    model.test(config, ur_fall_test_dataset)

show_msg('UP Fall test set results by clip')
model.test_by_clip(config, test_dataset)
show_msg('UR Fall test set results by clip')
model.test_by_clip(config, ur_fall_test_dataset)