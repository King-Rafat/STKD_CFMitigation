from torch.utils.data import Dataset, DataLoader
from skimage import io
import cv2
import glob
import random

import torchvision.transforms as transforms




class TinyImagenetDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, use_cache = True, cache_size = 50000 ,transform = None):
        self.image_paths = image_paths
        self.transform = transform
        self.cached_data = []
        self.cache_size = cache_size
        self.cache = {}
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if index in self.cache:
            image, label = self.cache[index]
        else:
            image_filepath = self.image_paths[index]
            image_filepath = image_filepath.replace('\\', '/')
            image = io.imread(image_filepath) # your slow data loading
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = image_filepath.split('/')[-2]
            label = self.class_to_idx[label]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (image, label)
        
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label