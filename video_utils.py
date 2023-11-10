import numpy as np
import cv2
import torch
import os


class Video():
    def __init__(self, filename):
        cap = cv2.VideoCapture(filename)
        self.filename = filename
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.time = self.length/self.fps
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("The file doesn't exist, or it's not a video")
            raise FileExistsError
        else:
            self.cap = cap
    
    
    def read_frame(self, n=-1, to_rgb=True):
        current_frame = self.__current_frame_number()
        if n != -1:
            self.__set_frame(n)
        cap = self.cap
        if cap.isOpened():
            _, frame = cap.read()
            if to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if n != -1:
                self.__set_frame(current_frame)
            return frame
        
    
    def __set_frame(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    
    def __current_frame_number(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    
    def return_list(self, total_frames=-1, startframe=0):
        current_frame = self.__current_frame_number()
        self.__set_frame(startframe)
        frames = [] 
        frame = self.read_frame()
        while True:
            frames.append(frame)
            if self.__current_frame_number() == self.length:
                break
            frame = self.read_frame()
        self.__set_frame(current_frame)
        if total_frames != -1:
            frames = frames[1::int(np.ceil(self.length / total_frames))]
        return frames
    

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, num_frames):
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        for c in classes:
            self.video_paths.extend([os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c))])
            self.labels.extend([c for _ in range(len(os.listdir(os.path.join(root, c))))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vid = Video(self.video_paths[idx]) #.permute(2, 0, 1) / 255
        frames = np.array(vid.return_list(total_frames=self.num_frames))
        if len(frames) != self.num_frames:
            print(f'len mismatch: {len(frames)} vs {self.num_frames}')
        print(frames.shape)
        # frames = frames.mean(axis=0) / 255
        label = self.labels[idx]
        return (frames, label)
