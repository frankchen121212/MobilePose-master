import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

from utils.dataset_utils import *
from skimage.io import imread
from scipy.io import loadmat


class LSP(Dataset):
    def __init__(self,
                 T=5,
                 root='data/LSP',
                 transformer=None,
                 train=True,
                 output_size=256,
                 label_size=31,
                 sigma_center=21,
                 sigma_label=2):

        self.T = T
        self.root = root
        self.train = train
        self.output_size = output_size
        self.transformer = transformer
        self.sigma_center = sigma_center
        self.sigma_label = sigma_label
        self.label_size = label_size

        annotations_label = 'train_' if train else 'valid_'
        self.annotations_path = os.path.join(self.root, annotations_label + 'annotations.json')

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations(train)

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = self.annotations[str(idx)]['image_path']
        image = imread(path).astype(np.float32)
        x, y, visibility = self.load_annotation(idx)
        image, x, y, visibility,bbox, unnormalized = self.transformer(image, x, y, visibility,bbox=None)

        label_map = compute_label_map(x, y, self.output_size, self.label_size, self.sigma_label)
        center_map = compute_center_map(x, y, self.output_size, self.sigma_center)
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        meta = torch.from_numpy(np.squeeze(np.hstack([x, y]))).float()
        image = torch.unsqueeze(image, 0).repeat(self.T, 1, 1, 1)
        unnormalized = torch.unsqueeze(unnormalized, 0).repeat(self.T, 1, 1, 1)
        label_map = label_map.repeat(self.T, 1, 1, 1)
        return image, label_map, center_map, meta, unnormalized,visibility,bbox

    def load_annotation(self, idx):
        labels = self.annotations[str(idx)]['joints']
        x, y, visibility = self.dict_to_numpy(labels)
        x, y, visibility = self.reorder_joints(x, y, visibility)
        for i_joint in range(visibility.shape[0]):
            if int(visibility[i_joint]) == 0:
                visibility[i_joint] = 1
            else:
                visibility[i_joint] = 0
        return x, y, visibility

    def generate_annotations(self,train):
        annotation_path = os.path.join(self.root, 'joints.mat')
        image_root = os.path.join(self.root, 'images')
        start_index = 0
        if not train:
            start_index = 1000
        data = {}
        i = 0
        annotations = loadmat(annotation_path)['joints']

        for j in range(start_index, start_index+1000):
            length = 4
            image_path = os.path.join(image_root, 'im' + str(j + 1).zfill(length) + '.jpg')
            joints = annotations[:, :, j]
            x   = joints[0, :]
            y   = joints[1, :]
            vis = joints[2, :]
            joints_dict = {}
            for p_id, (p_x, p_y, p_vis) in enumerate(zip(x, y, vis)):
                joints_dict[str(p_id)] = (p_x, p_y, int(p_vis))

            data[i] = {'image_path': image_path,
                       'joints': joints_dict}
            i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)

    @staticmethod
    def reorder_joints(x, y, vis):
        mpii_order = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9]
        return x[mpii_order], y[mpii_order], vis[mpii_order]

    @staticmethod
    def dict_to_numpy(data):
        n = len(data)
        x, y, vis = np.zeros(n), np.zeros(n), np.zeros(n)
        for p in range(n):
            x[p] = data[str(p)][0]
            y[p] = data[str(p)][1]
            vis[p] = data[str(p)][2]
        return x, y, vis
