
from torch.utils.data import Dataset

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
            self.generate_annotations()

        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = self.annotations[str(idx)]['image_path']
        image = imread(path).astype(np.float32)
        x, y, visibility = self.load_annotation(idx)

        if self.transformer is not None:
            image, x, y, visibility,bbox, unnormalized = self.transformer(image, x, y, visibility,bbox=None)
        else:
            bbox = None

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

    def generate_annotations(self):
        data = {}
        i = 0

        annotation_paths = [os.path.join(self.root, 'joints_2000.mat'),
                            os.path.join(self.root, 'joints_10000.mat')]
        image_roots = [os.path.join(self.root, 'images_2000'),
                       os.path.join(self.root, 'images_10000')]

        for annotation_path, image_root in zip(annotation_paths, image_roots):
            annotations = loadmat(annotation_path)['joints']
            length = 5 if '10000' in annotation_path else 4
            if '10000' in annotation_path:
                annotations = np.moveaxis(annotations, 2, 0)
            else:
                annotations = np.moveaxis(annotations, (0, 1, 2), (2, 1, 0))

            for j in range(annotations.shape[0]):
                train = '10000' in annotation_path or j < 1000
                if train == self.train:
                    image_path = os.path.join(image_root, 'im' + str(j + 1).zfill(length) + '.jpg')
                    joints = annotations[j, :, :]
                    x, y, vis = np.split(joints, 3, axis=1)
                    x, y, vis = np.squeeze(x, axis=1), np.squeeze(y, axis=1), np.squeeze(vis, axis=1)

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
