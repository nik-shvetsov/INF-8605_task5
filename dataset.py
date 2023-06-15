import os
from glob import glob
from pathlib import Path
import config

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import lovely_tensors as lt
lt.monkey_patch()


class Food11Dataset(Dataset):
    """
    Dataset class for Food-11 dataset
    
    https://www.kaggle.com/datasets/vermaavi/food11

    This is a dataset containing 16643 food images grouped in 11 major categories

    Labels: {'bread': 0, 'dairy_product': 1, 'dessert': 2, 'egg': 3, 'fried_food': 4, 'meat': 5, 'noodles_pasta': 6, 'rice': 7, 'seafood': 8, 'soup': 9, 'vegetable_fruit': 10}
    """
    def __init__(self, data_path, tf, preproc=True, augment=True):
        self.data_path = data_path

        classes = ["bread", "dairy_product", "dessert", "egg", "fried_food", 
                   "meat", "noodles_pasta", "rice", "seafood", "soup", "vegetable_fruit"]
        self.class_to_idx = {class_label: idx  for idx, class_label in enumerate(classes)}
        # {'bread': 0, 'dairy_product': 1, 'dessert': 2, 'egg': 3, 'fried_food': 4, 'meat': 5, 'noodles_pasta': 6, 'rice': 7, 'seafood': 8, 'soup': 9, 'vegetable_fruit': 10}
        
        self.idx_to_class = {idx: class_label  for idx, class_label in enumerate(classes)}
        # {0: 'bread', 1: 'dairy_product', 2: 'dessert', 3: 'egg', 4: 'fried_food', 5: 'meat', 6: 'noodles_pasta', 7: 'rice', 8: 'seafood', 9: 'soup', 10: 'vegetable_fruit'}
        
        self.imgs = sorted(glob(f"{os.path.join(data_path)}/*.jpg"))
        self.targets = [int(Path(img).name.split('_')[0]) for img in self.imgs]
        self.target_names = [self.idx_to_class[int(Path(img).name.split('_')[0])] for img in self.imgs]

        self.transforms = tf
        self.augment = augment
        self.preproc = preproc

        self.imgs_per_class = {}
        for class_idx in self.class_to_idx.values():
            self.imgs_per_class[class_idx] = len(glob(f"{os.path.join(data_path)}/{class_idx}_*"))
        
        assert len(classes) == config.NUM_CLASSES, f"Number of classes in dataset ({len(classes)}) does not match config ({config.NUM_CLASSES})"
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        preproc_img = {'image': cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)}
        if self.preproc:
            preproc_img['image'] = Image.fromarray(preproc_img['image'])
            preproc_img['image'] = self.transforms['preproc'](preproc_img['image'])
            preproc_img['image'] = preproc_img['image'].permute(1, 2, 0)
            preproc_img['image'] = np.asarray(preproc_img['image'])
        if self.augment:
            preproc_img = self.transforms['aug'](image=preproc_img['image'])

        preproc_img = self.transforms[f'resize_to_tensor'](image=preproc_img['image'])
        img = preproc_img['image']
        
        label = torch.tensor(int(Path(self.imgs[idx]).name.split('_')[0]))
        return img, label
            

if __name__ == '__main__':
    test_dataset = Food11Dataset(config.TEST_DIR, config.ATF, preproc=False, augment=False)
    dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    n_show = 4

    print("=====================================")
    print("Dataset info:")
    print(f"Labels: ", test_dataset.class_to_idx)
    print(f"Number of images per class: ", test_dataset.imgs_per_class)
    print("=====================================")

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        print(f"Imgs tensor: {imgs}")
        print(f"Image labels: {labels[:n_show]}")
        x = imgs[:n_show] if n_show < config.BATCH_SIZE else imgs[:config.BATCH_SIZE]
        grid = torchvision.utils.make_grid(x.view(-1, 3, config.INPUT_SIZE[0], config.INPUT_SIZE[1]))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig('dataloader_pbatch.png')
        break
