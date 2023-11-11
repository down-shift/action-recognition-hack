import numpy as np
import cv2
import torch
import os
from torch.utils.data import Dataset


def read_video(path, frames_num=8):
    frames = []
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length//(frames_num)
    assert N > 0, 'Too many frames requested'
    current_frame = 0
    for i in range(length):
        ret, frame = cap.read(current_frame)
        if ret and i == current_frame and len(frames) < frames_num:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            current_frame += N
    cap.release()
    return frames


class VideoDataset(Dataset):
    def __init__(self, root, classes, num_frames, transform=None, mean_frames=False):
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.mean_frames = mean_frames
        for idx, c in classes.items():
            self.video_paths.extend([os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c))])
            self.labels.extend([idx for _ in range(len(os.listdir(os.path.join(root, c))))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frames = read_video(self.video_paths[idx], frames_num=self.num_frames)
        # frames = np.array(vid.return_list(total_frames=self.num_frames))
        frames = torch.tensor(np.array(frames))
        if len(frames) != self.num_frames:
            print(f'len mismatch: {len(frames)} vs {self.num_frames}')
        
        if self.mean_frames:
            frames = torch.mean(frames.to(torch.float32), axis=0)
        if self.transform:
            frames = self.transform(frames)
        label = self.labels[idx]
        return (frames, label)
