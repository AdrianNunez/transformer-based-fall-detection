import csv
import glob
import torch
import numpy as np
from scipy import stats
from app import load_video
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

def load_labels(config):
    labels_file = config["labels_file"]
    fall_classes = config["fall_classes"]
    train_subjects = config["train_subjects"]
    dev_subjects = config["dev_subjects"]
    test_subjects = config["test_subjects"]
    window_size = config["window_size"]
    binarise = config["binarise"]
    competition_split = config["competition_split"]
    train, dev, test = [], [], []
    with open(labels_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        frame_no = 0
        video, clip = [], []
        for row in csv_reader:
            if line_count > 1:
                subject = int(row[-4])
                activity = int(row[-3])
                trial = int(row[-2])
                # At the beginning, store the identification
                if len(video) == 0:
                    previous_frame = (subject, activity, trial)
                # If a new sequence, different from the previous, starts
                # store the video so far
                else:
                    if previous_frame != (subject, activity, trial):
                        # If the video is not divisible by window_size
                        # i.e. the last clip does not have size window_size
                        if len(clip) > 0 and len(clip) < window_size:
                            diff = window_size - len(clip)
                            last_entry = clip[-1]
                            for _ in range(diff): clip.append(last_entry)
                            clip_label = -1
                            if binarise:
                                clip_label = 0
                                for _,_,_,_,label in clip:
                                    if label == 1: 
                                        clip_label = 1
                                        break
                            else:
                                clip_labels = [label for _,_,_,_,label in clip]
                                clip_label = stats.mode(
                                    clip_labels, axis=None
                                ).mode[0]
                            video.append([previous_frame, clip_label, clip])
                            clip = []


                        if competition_split:
                            if previous_frame[0] in dev_subjects and previous_frame[2] == 3:
                                dev.append(video)
                            elif previous_frame[0] in test_subjects:
                                test.append(video)
                            elif previous_frame[0] in train_subjects:
                                train.append(video)
                        else:
                            if previous_frame[2] == 3:
                                test.append(video)
                            # DEV: dev_subjects and trial 2
                            elif previous_frame[0] in dev_subjects and previous_frame[2] == 2:
                                dev.append(video)
                            # TRAIN: rest
                            elif previous_frame[0] in train_subjects:
                                train.append(video)
                        # Reset the video array
                        video = []
                        previous_frame = (subject, activity, trial)
                        frame_no = 0
              
                # `activity_label` can have 11 values
                activity_label = int(row[-1])
                if activity_label == 20:
                    continue
                if binarise:
                    # reduce it to a binary fall/not fall label
                    label = 0
                    if activity_label in fall_classes:
                        label = 1
                else:
                    label = activity_label-1

                clip.append([subject, activity, trial, frame_no, label])
                if len(clip) == window_size:
                    clip_label = -1
                    if binarise:
                        clip_label = 0
                        for _,_,_,_,label in clip:
                            if label == 1: 
                                clip_label = 1
                                break  
                    else:
                        clip_labels = [label for _,_,_,_,label in clip]
                        clip_label = stats.mode(clip_labels, axis=None).mode[0]
                    video.append([previous_frame, clip_label, clip])
                    clip = []

                frame_no += 1
            else:
                line_count += 1
    print('Train videos: {}'.format(len(train)))
    print('Dev videos: {}'.format(len(dev)))
    print('Test videos: {}'.format(len(test)))
    
    return train, dev, test

class FallDataset(Dataset):
    def __init__(self, config, labels, mode):
        self.mode = mode
        self.dataset_folder = config["dataset_folder"]
        batch_size = config["batch_size_{}".format(mode)]
        oversample = config["oversample"]
        undersample = config["undersample"]
        binarise = config["binarise"]
        weighted_sampler = config["weighted_sampler"]

        nb_classes = len(config["class_names"])
        if binarise: nb_classes = 2
      
        # store the dataset in an array format for __getitem__
        if mode == 'test':
            # a video per sample
            self.labels = labels 
        else:
            # a clip per sample
            self.labels = [clip for video in labels for clip in video]
            if mode == 'train':
                if oversample:
                    samples_by_class = [[] for _ in range(nb_classes)]
                    #fall_samples = []
                    #nb_no_fall = 0
                    for label in self.labels:
                        (_, _, _), clip_label, _ = label
                        samples_by_class[clip_label].append(label)
                
                    #print(nb_no_fall, len(fall_samples))
                    max_samples = np.max([len(x) for x in samples_by_class])
                    for c in range(nb_classes):
                        diff = max_samples - len(samples_by_class[c])
                        if diff == 0: continue
                        idx = np.random.choice(range(len(samples_by_class[c])), size=diff)
                        new_samples = [samples_by_class[c][i] for i in idx]
                        self.labels.extend(new_samples)
                   
                    del samples_by_class
                elif undersample:
                    no_fall_samples, fall_samples = [], []
                    for label in self.labels:
                        (_, _, _), clip_label, _ = label
                        if clip_label == 1:
                            fall_samples.append(label)
                        else:
                            no_fall_samples.append(label)
                    indices = np.random.choice(
                        range(len(no_fall_samples)), 
                        len(fall_samples)
                    )
                    no_fall_samples = [no_fall_samples[i] for i in indices]
                    self.labels = fall_samples + no_fall_samples

        # Generate weights for each input (for weighted sampling)
        sampler = None
        if mode == 'train' and weighted_sampler and binarise:
            # Shuffle
            indices = np.random.permutation(range(len(self.labels)))
            self.labels = [self.labels[i] for i in indices]

            self.sample_weights = []
            nb_no_fall, nb_fall = 0, 0
            for label in self.labels:
                (_, _, _), clip_label, _ = label
                if clip_label == 0: nb_no_fall += 1
                else: nb_fall += 1
 
            no_fall_prob = nb_no_fall / (nb_no_fall+nb_fall)
            fall_prob = 1-no_fall_prob
            
            for label in self.labels:
                (_, _, _), clip_label, _ = label
                if clip_label == 0: self.sample_weights.append(no_fall_prob)
                else: self.sample_weights.append(fall_prob)

            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.labels)
            )

        # Function used to create batches
        def collate_fn(data):
            videos, labels = [], []
            for sample in data:
                videos.append(sample['data'])
                labels.append(sample['label'])

            if mode == 'test':
                videos = pad_sequence(videos)
                labels = pad_sequence(labels, padding_value=-1)
            else:
                videos = torch.cat(videos)
                labels = torch.as_tensor(labels)

            return {
                'data': videos,
                'label': labels
            }
       
        # Object in charge of managing batches
        self.dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=True if mode == 'train' and sampler is None else False,
            collate_fn=collate_fn,
            sampler=sampler if mode == 'train' else None,
            num_workers=4,
            drop_last=True,
            prefetch_factor=8
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.mode == 'test':
            video = self.labels[i]
            subject, activity, trial = video[0][0][0], video[0][0][1], video[0][0][2]
        else:
            (subject, activity, trial), clip_label, clip = self.labels[i]
        
        sub_path = self.dataset_folder + \
            'Subject{}/Activity{}/Trial{}/Camera1/frame_'.format(
                subject,
                activity,
                trial
            )

        # TEST
        if self.mode == 'test':
            loaded_videos, clip_labels = [], []
            for clip_container in video:
                (_, _, _), clip_label, clip = clip_container
                frames = sorted(
                    [sub_path + '{:04d}.png'.format(n) for _,_,_,n,_ in clip]
                )
                vid = load_video(frames)
                loaded_videos.append(vid)
                clip_labels.append(clip_label)
           
            return {
                'data': torch.cat(loaded_videos),
                'label': torch.as_tensor(clip_labels)
            }
        # TRAIN / VALIDATION
        else:
            frames = sorted(
                [sub_path + '{:04d}.png'.format(n) for _,_,_,n,_ in clip]
            )

            vid = load_video(frames)
           
            return {
                'data': vid,
                'label': clip_label
            }