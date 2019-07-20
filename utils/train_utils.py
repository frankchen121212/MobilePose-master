
import torch.nn as nn
import os
import torch
import shutil


def save_checkpoint(state, is_best, checkpoint,dataset_name,log, prefix=''):
    filepath = os.path.join(checkpoint,dataset_name, prefix + 'last.pth.tar')
    file_root = os.path.join(checkpoint,dataset_name)

    if not os.path.exists(file_root):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        log.info("Saving checkpoint... ")
    torch.save(state, filepath)
    if is_best:
        log.info("Checkpoint is best!")
        shutil.copyfile(filepath, os.path.join(checkpoint,dataset_name, prefix + 'best.pth.tar'))


def load_checkpoint(model_dir,dataset_name, checkpoint_name, model, optimizer=None):
    path = os.path.join(model_dir,dataset_name, checkpoint_name)
    if not os.path.exists(path):
        print(path)
        raise('File does not exist {}'.format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint


def initialize_weights_normal(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def initialize_weights_kaiming(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_parameters_rec(model):
    return sum(p.numel() for module in model.modules() for p in module.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)
