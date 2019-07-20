from torch.utils.data import Dataset
import sys
import os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import os
import json
import numpy
from utils.dataset_utils import *
from utils.augmentation import *
from skimage.io import imread
from scipy.io import loadmat

########################################################################
#0 - r ankle, 1 - r knee, 2 - r hip,3 - l hip,4 - l knee,
# 5 - l ankle, 6 - l ankle， 7 - l ankle，8 - upper neck, 9 - head top,
# 10 - r wrist,11 - r elbow, 12 - r shoulder, 13 - l shoulder,14 - l elbow, 15 - l wrist                                                                      #
#

class MPII(Dataset):
    def __init__(self,
                 T=5,
                 root='data/MPII',
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
        self.sigma_center = sigma_center
        self.sigma_label = sigma_label
        self.label_size = label_size
        self.transformer = transformer
        self.pixel_std = 200

        annotations_label = 'train' if train else 'valid'
        self.annotations_path = os.path.join(self.root, annotations_label + '.json')

        if not os.path.isfile(self.annotations_path):
            self.generate_annotations()
        with open(self.annotations_path) as data:
            self.annotations = json.load(data)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        path = os.path.join(self.root,'images',self.annotations[idx]['image'])
        #frames = self.load_video(idx)
        image = imread(path).astype(np.float32)
        width = image.shape[1]
        height = image.shape[0]
        x, y, visibility = self.load_annotation(idx)
        bbox ,center ,scale = self.load_bbox(idx,width,height)


        if self.transformer is not None:
            image, x, y, visibility,bbox, unnormalized = self.transformer(image, x, y, visibility,bbox)
        else:
            unnormalized = Transformer.to_torch(image)
            image = Transformer.to_torch(image)

        label_map = compute_label_map(x, y, self.output_size, self.label_size, self.sigma_label)

        center_map = compute_center_map(x, y, self.output_size, self.sigma_center)

        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        meta = torch.from_numpy(np.squeeze(np.hstack([x, y]))).float()

        image = torch.unsqueeze(image, 0).repeat(self.T, 1, 1, 1)
        unnormalized = torch.unsqueeze(unnormalized, 0).repeat(self.T, 1, 1, 1)
        label_map = label_map.repeat(self.T, 1, 1, 1)

        return image, label_map, center_map, meta, unnormalized ,visibility,bbox




    def load_bbox(self,idx,width,height):
        center = np.array(self.annotations[idx]['center'])
        scale = np.array(self.annotations[idx]['scale'])
        #[x_min, y_min, x_max, y_max]
        w = scale*self.pixel_std
        h = scale*self.pixel_std
        x_min = max(int(center[0] - w*0.5),0)
        x_max = min(int(center[0] + w*0.5),width)
        y_min = max(int(center[1] - h*0.5),0)
        y_max = min(int(center[1] + h*0.5),height)
        bbox = [x_min,y_min,x_max,y_max]
        return bbox,center,scale


    def load_annotation(self, idx):
        labels = self.annotations[idx]['joints']
        vis_labels = self.annotations[idx]['joints_vis']
        x_, y_ = self.dict_to_numpy(labels)
        visibility_ = self.vis_dict_to_numpy(vis_labels)
        x ,y,vis = [],[],[]
        ignored = [6, 7]  # Ignore pelvis and thorax
        shifted = [14, 15]  # Indices to replace pelvis and thorax
        for i in range(0,len(vis_labels)):
            if i in ignored:
                x.append(int(x_[i+8]))
                y.append(int(y_[i+8]))
                vis.append(visibility_[i+8])
            elif i in shifted:
                continue
            else:
                x.append(int(x_[i]))
                y.append(int(y_[i]))
                vis.append(visibility_[i])

        x = numpy.array(x)
        y = numpy.array(y)
        vis = numpy.array(vis)
        return x, y, vis


    def generate_annotations(self):
        data = {}
        i = 0

        annotations = loadmat(os.path.join(self.root, 'annotations.mat'))['RELEASE']

        for image_idx in range(annotations['img_train'][0][0][0].shape[0]):
            if self.train == self.is_train(annotations, image_idx):
                image_path = os.path.join(self.root, 'images', self.get_image_name(annotations, image_idx))
                for person_idx in range(self.n_people(annotations, image_idx)):
                    c, s = self.location(annotations, image_idx, person_idx)
                    if not c[0] == -1:
                        joints = self.get_person_joints(annotations, image_idx, person_idx)

                        if len(joints) > 0:
                            ignored = ['6', '7']  # Ignore pelvis and thorax
                            shifted = ['14', '15']  # Indices to replace pelvis and thorax

                            for idx_ignored, idx_shifted in zip(ignored, shifted):
                                joints[idx_ignored] = joints[idx_shifted]
                                del joints[idx_shifted]

                            data[i] = {'image_path': image_path,
                                       'joints': joints}
                            i += 1

        with open(self.annotations_path, 'w') as out_file:
            json.dump(data, out_file)

    @staticmethod
    def reorder_joints(x, y, vis):
        mpii_order = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9]
        return x[mpii_order], y[mpii_order], vis[mpii_order]

    @staticmethod
    def get_person_joints(annotations, image_idx, person_idx):
        mpii_joints = 16
        joints = {}
        image_info = annotations['annolist'][0][0][0]['annorect'][image_idx]
        if 'annopoints' in str(image_info.dtype) and image_info['annopoints'][0][person_idx].size > 0:
            person_info = image_info['annopoints'][0][person_idx][0][0][0][0]
            if len(person_info) == mpii_joints:
                for i in range(mpii_joints):
                    p_id, p_x, p_y = person_info[i]['id'][0][0], \
                                     int(person_info[i]['x'][0][0]),\
                                     int(person_info[i]['x'][0][0])
                    vis = 1
                    if 'is_visible' in person_info.dtype.fields:
                        vis = person_info[i]['is_visible']
                        vis = int(vis[0][0]) if len(vis) > 0 else 1

                    joints[str(p_id)] = (p_x, p_y, vis)
        return joints

    @staticmethod
    def get_image_name(annotations, image_idx):
        return str(annotations['annolist'][0][0][0]['image'][:][image_idx][0][0][0][0])

    @staticmethod
    def dict_to_numpy(data):
        n = len(data)
        x, y = np.zeros(n), np.zeros(n)
        for p in range(n):
            x[p] = data[p][0]
            y[p] = data[p][1]

        return x, y

    @staticmethod
    def vis_dict_to_numpy(data):
        n = len(data)
        vis = np.zeros(n)
        for p in range(n):
            vis[p] = data[p]
        return vis

    # Functions below taken from https://github.com/umich-vl/pose-hg-train/blob/master/src/misc/mpii.py
    @staticmethod
    def n_people(annot, image_idx):
        example = annot['annolist'][0][0][0]['annorect'][image_idx]
        if len(example) > 0:
            return len(example[0])
        else:
            return 0

    @staticmethod
    def is_train(annotations, image_idx):
        return (annotations['img_train'][0][0][0][image_idx] and
                annotations['annolist'][0][0][0]['annorect'][image_idx].size > 0 and
                'annopoints' in annotations['annolist'][0][0][0]['annorect'][image_idx].dtype.fields)

    @staticmethod
    def location(annot, image_idx, person_idx):
        example = annot['annolist'][0][0][0]['annorect'][image_idx]
        if ((not example.dtype.fields is None) and
                'scale' in example.dtype.fields and
                example['scale'][0][person_idx].size > 0 and
                example['objpos'][0][person_idx].size > 0):
            scale = example['scale'][0][person_idx][0][0]
            x = example['objpos'][0][person_idx][0][0]['x'][0][0]
            y = example['objpos'][0][person_idx][0][0]['y'][0][0]
            return np.array([x, y]), scale
        else:
            return [-1, -1], -1

def main():
    from utils.augmentation import ImageTransformer
    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    root = '../data/MPII'
    output_root = '../output/MPII'
    log_dir = '../logs'
    logger = SummaryWriter(log_dir)

    mean_path = os.path.join(root, 'means.npy')

    mean, std = np.load(mean_path)
    transformer = ImageTransformer
    train_transformer = transformer(output_size=256, mean=mean, std=std)
    loader_args = {'num_workers': 1, 'pin_memory': True}
    train_dataset = MPII(train=True, T=5, root=root, transformer=train_transformer,
                            output_size=256, sigma_center=21, sigma_label=2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False,
                              **loader_args)

    for i, (frames, labels, centers, meta, unnormalized) in enumerate(train_loader):
        frames, labels, centers, meta = frames.cuda(), labels.cuda(), centers.cuda(), meta.cuda()
        batch_size, n_stages, n_joints = labels.shape[0], labels.shape[1], labels.shape[2]

        debug_inputs(unnormalized,labels,centers,logger,output_root,i)


def debug_inputs(video, labels, center_map, logger,output_root,i_loader):
    # if video.shape[1] != outputs.shape[1]:
    #     f_0 = torch.unsqueeze(video[:, 0, :, :, :], 1)
    #     video = torch.cat([f_0, video], dim=1)
    #
    # if labels.shape[1] != outputs.shape[1]:
    #     l_0 = torch.unsqueeze(labels[:, 0, :, :, :], 1)
    #     labels = torch.cat([l_0, labels], dim=1)

    batch_size, n_stages, n_joints = labels.shape[0], labels.shape[1], labels.shape[2]

    video = video.detach().numpy().copy().astype(np.uint8)
    video = np.moveaxis(video, 2, 4)

    labels = labels.detach()
    #outputs = outputs.detach()

    image_size = video.shape[-2]
    label_size = labels.shape[-2]
    r = image_size / label_size

    for i in range(batch_size):
        batch_gt_coords = get_preds(labels[i, :, :, :, :]).cpu().numpy()  # add .cpu()
        #batch_pred_coords = get_preds(outputs[i, :, :, :, :]).cpu().numpy()  # add .cpu()

        for t in range(n_stages):
            frame = video[i, t, :, :, :]
            frame_gt_coords = np.flip(batch_gt_coords[t, :, :] * r, axis=1)
            #frame_pred_coords = np.flip(batch_pred_coords[t, :, :] * r, axis=1)

            gt_frame = draw_skeleton(frame.copy(), frame_gt_coords)
            #pred_frame = draw_skeleton(frame.copy(), frame_pred_coords)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            plot_matches(ax, gt_frame, gt_frame, frame_gt_coords, frame_gt_coords, np.array([]),
                         keypoints_color='red')
            plt.show()
            plt.savefig(os.path.join(output_root,'{}_stage_{}_.jpg'.format(i_loader,t)))



if __name__ == '__main__':
    main()
