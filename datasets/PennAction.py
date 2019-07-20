import os
from torch.utils.data import Dataset
import sys
import scipy.io as scio

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import torch
import math
import six
import time
import lmdb
import numpy as np
from PIL import Image
from utils.dataset_utils import *
from utils.vis import save_visualize_result
from utils.evaluation import get_preds
# from skimage.io import imread
from scipy.io import loadmat

class PennAction(Dataset):
    def __init__(self,
                 T=5,
                 root='data/PennAction',
                 output_root='output',
                 transformer=None,
                 train=True,
                 output_size=256,
                 label_size=31,
                 sigma_center=21,
                 sigma_label=2):

        self.T = T
        self.root = root
        self.output_root = output_root
        self.train = train
        self.output_size = output_size
        self.transformer = transformer
        self.sigma_center = sigma_center
        self.sigma_label = sigma_label
        self.label_size = label_size


        annotations_label = 'train_' if train else 'valid_'
        self.annotations_path = os.path.join(self.root, annotations_label + 'annotations_' + str(self.T) + '.npy')

        if not os.path.isfile(self.annotations_path):
            print('annotation file: {} not found, creating...'.format(self.annotations_path))
            self.generate_annotations()

        print('loading annotation file: {} ...'.format(self.annotations_path))
        self.annotations = np.load(self.annotations_path)[()]
        print('annotation file loaded!')
        data_label = 'train.lmdb' if train else 'valid.lmdb'
        self.data_path = os.path.join(self.root, data_label)

        print('loading dataset: {} ...'.format(self.data_path))
        self.env = lmdb.open(self.data_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        print('data loaded!')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        frames = self.load_video(idx)
        x, y, visibility, bbox,annotations_path,start_index = self.load_annotation(idx)
        raw_x = x.copy()
        raw_y = y.copy()
        
        if self.transformer is not None:
            transformed_frames, x, y, flipped_v, bbox, unnormalized,raw_frames = self.transformer(frames, x, y, visibility, bbox)
        else:
            transformed_frames = frames
            flipped_v = visibility


        label_map = compute_label_map(x, y, self.output_size, self.label_size, self.sigma_label)
        center_map = compute_center_map(x, y, self.output_size, self.sigma_center)
        meta = torch.from_numpy(np.squeeze(np.hstack([x,y]))).float()
        flipped_v = torch.from_numpy(flipped_v)
        bbox = torch.from_numpy(bbox.astype(np.float32))
        #return transformed_frames, label_map, center_map, meta, unnormalized,raw_frames,raw_x,raw_y,raw_v,flipped_v
        return transformed_frames, label_map, center_map,meta, unnormalized ,flipped_v,bbox

    def load_annotation(self, idx):
        # annotations_path = self.annotations[str(idx)]['annotations_path']
        # start = int(self.annotations[idx]['start_index'])
        annotations = self.annotations[idx]
        annotations_path = annotations['annotations_path']
        start_index = int(annotations['start_index'])
        # x = annotations['x']
        # y = annotations['y']
        vis = annotations['visibility']
        bbox = annotations['bbox']
        # Read the .mat file
        label = scio.loadmat(os.path.join(annotations_path))

        x_raw = label['x'][start_index:start_index + self.T, :]
        y_raw = label['y'][start_index:start_index + self.T, :]
        vis_raw = label['visibility'][start_index:start_index + self.T, :]
        x_raw, y_raw, vis_raw = self.infer_neck_annotation(x_raw, y_raw, vis_raw)
        x_raw, y_raw, vis_raw = self.reorder_joints(x_raw, y_raw, vis_raw)

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)
        #
        # print('x_raw--{}'.format(x_raw))

        # x, y, vis = self.infer_neck_annotation(x, y, vis)
        # x, y, vis = self.reorder_joints(x, y, vis)

        return x_raw, y_raw, vis, bbox ,annotations_path,start_index

    def load_video(self, idx):
        video = self.annotations[idx]['frames_root'].split('/')[-2]
        start = int(self.annotations[idx]['start_index'])
        frame_paths = self.index_to_path(video, start, self.T)
        frames = []
        with self.env.begin(write=False) as txn:
            for img_path in frame_paths:
                # print(img_path)
                imgbuf = txn.get(img_path.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                img = np.array(Image.open(buf).convert('RGB'))
                frames.append(img)
        return frames

    def index_to_path(self, folder, idx, T):
        paths = []
        T = int(T)
        for i in range(T):
            path = folder + '/' + str(idx + i + 1).zfill(6) + '.jpg'
            paths.append(path)
        return paths

    def generate_annotations(self):
        data = {}
        i = 0
        annotations_directory = os.path.join(self.root, 'labels')
        for file in os.listdir(annotations_directory):
            filename = os.fsdecode(file)
            if filename.endswith('.mat'):
                annotation_path = os.path.join(annotations_directory, filename)
                annotations = loadmat(annotation_path)
                video_id = filename.split('.')[0]
                frames_root = os.path.join(self.root, 'frames', video_id, '*')
                n = annotations['nframes']
                train = bool(annotations['train'][0][0] + 1)
                if train == self.train:
                    if n % self.T == 0:
                        indices = np.arange(0, n, self.T)
                    else:
                        indices = np.arange(0, n, self.T)[:-1]   # Exclude last range, may not be `T` long
                    for start_index in indices:
                        x = annotations['x'][start_index:start_index + self.T, :]
                        y = annotations['y'][start_index:start_index + self.T, :]
                        vis = annotations['visibility'][start_index:start_index + self.T, :]
                        bbox = annotations['bbox'][start_index:start_index + self.T, :]
                        x, y, vis = self.infer_neck_annotation(x, y, vis)
                        x, y, vis = self.reorder_joints(x, y, vis)

                        data[i] = {'annotations_path': annotation_path,
                                   'frames_root': frames_root,
                                   'start_index': str(start_index),
                                   'x': x,
                                   'y': y,
                                   'visibility': vis,
                                   'bbox': bbox}
                        i += 1

        np.save(self.annotations_path, data)

    @staticmethod
    def infer_neck_annotation(x, y, vis):
        neck_x = np.expand_dims(0.5 * x[:, 0] + 0.25 * (x[:, 1] + x[:, 2]), 1)
        neck_y = np.expand_dims(0.2 * y[:, 0] + 0.4 * (y[:, 1] + y[:, 2]), 1)
        neck_vis = np.expand_dims(np.floor((vis[:, 0] + vis[:, 1] + vis[:, 2]) / 3.), 1)

        x = np.hstack([x, neck_x])
        y = np.hstack([y, neck_y])
        vis = np.hstack([vis, neck_vis])
        return x, y, vis

    @staticmethod
    def reorder_joints(x, y, vis):
        mpii_order = [12, 10, 8, 7, 9, 11, 3, 5, 13, 0, 6, 4, 2, 1]
        return x[:, mpii_order], y[:, mpii_order], vis[:, mpii_order]
########################################################################
#0 - r ankle, 1 - r knee, 2 - r hip,3 - l hip,4 - l knee,
# 5 - l ankle, 6 - l ankle， 7 - l ankle，8 - upper neck, 9 - head top,
# 10 - r wrist,11 - r elbow, 12 - r shoulder, 13 - l shoulder,14 - l elbow, 15 - l wrist                                                                      #
#



def to_numpy(tensor, scale=1.0, dtype=np.uint8):
    #tensor = tensor.cpu().detach().mul(scale).numpy().astype(dtype)
    tensor = tensor.cpu().detach().mul(scale).permute(1, 2, 0).numpy().astype(dtype)# add .cpu()
    return np.moveaxis(tensor, 0, 2)


def main():
    from utils.augmentation import VideoTransformer
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    root = '../data/PennAction'
    output_root = '../output/PennAction'
    mean_path = os.path.join(root, 'means.npy')
    mean, std = np.load(mean_path)
    transformer = VideoTransformer
    train_transformer = transformer(output_size=256, mean=mean, std=std)
    loader_args = {'num_workers': 1, 'pin_memory': True}
    train_dataset = PennAction(train=True, T=5, root=root, transformer=train_transformer,
                            output_size=256, sigma_center=21, sigma_label=2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False,
                              **loader_args)

    for i, (frames, labels, centers, meta, unnormalized,raw_frames,x_raw,y_raw,v_raw,vis_flipped) in enumerate(train_loader):
        frames, labels, centers, meta = frames.cuda(), labels.cuda(), centers.cuda(), meta.cuda()
        batch_size, n_stages, n_joints = labels.shape[0], labels.shape[1], labels.shape[2]
        # frames.shape ([1, 5, 3, 256, 256])
        # labels.shape ([1, 5, 15, 31, 31])
        # centers.shape ([1, 1, 256, 256])
        # raw_frames.shape([1, 5, 3, 256, 256])

        for j in range(batch_size):


            raw_image = raw_frames[0].clone().cpu()
            batch_image = frames[0].clone().cpu()
            batch_labels = get_preds(labels[j, :, :, :, :]).cpu().numpy()
            flipped_labels = meta.cpu().numpy() #[1,5,28]
            save_visualize_result(batch_image, labels, batch_labels, raw_image, flipped_labels, x_raw,
                                  y_raw, v_raw, vis_flipped, output_root, i, j)



            for t in  range(n_stages):
                current_frame = to_numpy(frames[j,t,:,:,:])
                current_label = to_numpy(labels[j, t, :, :, :], scale=255)
                current_label = np.maximum.reduce(current_label, axis=2)
                current_label = np.expand_dims(current_label, 2)
                current_label = cv2.cvtColor(current_label, cv2.COLOR_GRAY2RGB)
                current_label = cv2.resize(current_label, (256, 256))

                #inputs = cv2.addWeighted(current_frame, 0.3, current_label, 0.7, 0)



        # print('labels:{}'.format(labels))
        # print('meta:{}'.format(meta))
        time.sleep(10)


if __name__ == '__main__':
    main()
