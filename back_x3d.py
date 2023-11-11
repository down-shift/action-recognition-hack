import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import os
from torch.utils.data import Dataset
import warnings
import pandas as pd
import fvcore

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# функция для чтения видео
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


# класс тестового датасета
class VideoDataset(Dataset):
    def __init__(self, root, classes, num_frames, transform=None, mean_frames=False):
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.mean_frames = mean_frames
        for idx, c in classes.items():
            self.video_paths.extend([os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frames = read_video(self.video_paths[idx], frames_num=self.num_frames)
        frames = torch.tensor(np.array(frames))
        if len(frames) != self.num_frames:
            print(f'len mismatch: {len(frames)} vs {self.num_frames}')
        
        if self.mean_frames:
            frames = torch.mean(frames.to(torch.float32), axis=0)
        if self.transform:
            frames = self.transform(frames)
        return (frames, self.video_paths[idx])

# функция для выбора топ 5 классов по предсказанию модели
def get_top_props(inputs):
    top_probs = {}
    with torch.no_grad():
        logits = model(inputs.unsqueeze(0))
    probs = logits.softmax(axis=1).cpu()[0]
    s_probs = list(sorted(probs, reverse=True))
    mask = probs > s_probs[6]
    for i, flag in enumerate(mask):
        if flag:
            top_probs[CLASSES[i]] = probs[i].item()
    return top_probs

# функция для предобработки фреймов
def get_frames(path):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.),
        transforms.Lambda(lambda x: x.permute(3, 0, 1, 2)),  # (T, C, H, W)
        transforms.Lambda(lambda x: torch.nn.functional.interpolate(x, (224, 224)))
    ])
    video = read_video(path, frames_num=16)
    video = torch.tensor(np.array(video))
    video = transform(video.to(device))
    return video
    


if __name__ == "__main__":
    # configs:
    DATA_DIR = "Day_19__100_Pushups_a_Day_Challenge!_(_Break_Your_Rules_)_pushup_f_nm_np1_fr_goo_2.avi"  # путь до видео
    WEIGHTS = 'x3d_m_ep4_0.8162.pt' # путь до весов
    CLASSES = {0: 'cartwheel',1: 'catch', 2: 'clap', 3: 'climb', 4: 'dive', 5: 'draw_sword', 6: 'dribble', 7: 'fencing', 8: 'flic_flac', 9: 'golf', 10: 'handstand', 11: 'hit', 12: 'jump', 13: 'pick', 14: 'pour', 15: 'pullup', 16: 'push', 17: 'pushup', 18: 'shoot_ball', 19: 'sit', 20: 'situp', 21: 'swing_baseball', 22: 'sword_exercise', 23: 'throw'}
    # все наши классы
    SEED = 42
    
    warnings.filterwarnings("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    model.blocks[5].proj = nn.Linear(in_features=2048, out_features=24, bias=True)
    model.load_state_dict(torch.load(WEIGHTS, map_location=torch.device(device)))
    model = model.to(device)
    
    video = get_frames(DATA_DIR)
    print(get_top_props(video))