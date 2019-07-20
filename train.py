
import sys
import time
import argparse
import pycrayon
import configargparse
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datasets.MPII import MPII
from datasets.LSP import LSP
from datasets.PennAction import PennAction  # important

from torch.utils.data import DataLoader

from utils.train_utils import *
from utils.augmentation import *
from utils.dataset_utils import *
from utils.evaluation import accuracy
# from utils.parallel import DataParallelModel, DataParallelCriterion

from models.RecurrentStackedHourglass import PretrainRecurrentStackedHourglass
#from models.LSTMPoseMachine import LPM
from models.LSTMPoseMachine import LPM
from models.LSTMPoseMachine_weight_not_shared import LPM_raw


from models.CPM import CPM
from models.CoordinatePoseMachine import CoordinateLPM
from models.modules.ResidualBlock import ResidualBlock
from models.modules.InvertedResidualBlock import InvertedResidualBlock
from models.modules.ConvolutionalBlock import ConvolutionalBlock
from models.losses.MSESequenceLoss import MSESequenceLoss


from utils.log_creater import log_creater
from tensorboardX import SummaryWriter
from valid import validate

def train(model, loader, criterion, optimizer, logger, log, log_freq, device, r,output_root,i_epoch, scheduler=None, clip=None, summary=None, debug=False):
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    time_avg = RunningAverage()
    eval_dic_avg = RunningAverage()
    model.train()

    with tqdm(total=len(loader)) as t:
        for i, (frames, labels, centers,meta, unnormalized ,visibility,bbox) in enumerate(loader):
            frames, labels, centers,meta= frames.cuda(), labels.cuda(), centers.cuda(),meta.cuda()
            start = time.time()
            outputs ,outputs_loss = model(frames, centers)
            time_avg.update(time.time() - start)
            loss = criterion(outputs_loss, labels)
            acc,eval_dic = accuracy(outputs, labels, visibility ,r=r)
            eval_dic_avg.update(eval_dic)

            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            loss_avg.update(loss.item())
            acc_avg.update(acc)

            if debug is True and i%log_freq == 1:
                debug_predictions(unnormalized, labels, outputs, visibility, output_root,i,i_epoch)

            logger.add_scalar('Train Accuracy', acc, i)
            logger.add_scalar('Train Loss', loss.item(), i)

            t.set_postfix(loss='{:05.3f}'.format(loss.item()), acc='{:05.3f}%'.format(acc * 100),
                          loss_avg='{:05.3f}'.format(loss_avg()), acc_avg='{:05.3f}%'.format(acc_avg() * 100),
                          time_avg='{}ms'.format(int(1000 * time_avg())))
            t.update()

        return loss_avg(), acc_avg()

def main(args):
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    n_joints = 14
    start_time_prefix = time.strftime('%Y-%m-%d-%H-%M') + "_"
    print('Checkpoint prefix will be ' + start_time_prefix)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpus = [int(i) for i in args.gpus.split(',')]
    num_devices = len(gpus)
    device_name = 'cpu' if args.gpus is None else 'cuda:0'

    device = torch.device(device_name)

    loader_args = {'num_workers': 16, 'pin_memory': True} if 'cuda' in device_name else {}

    ''' tensorboard logger '''
    if not os.path.exists(args.log_dir):
        print('log_dir: {} not exists, creating...'.format(args.log_dir))
        os.makedirs(args.log_dir)

    logger = SummaryWriter(args.log_dir)

    if args.dataset == 'PennAction':
        dataset = PennAction
        transformer = VideoTransformer

    elif args.dataset == 'MPII':
        dataset = MPII
        transformer = ImageTransformer
    else:
        dataset = LSP
        transformer = ImageTransformer

    if not os.path.exists(os.path.join('output',args.model,args.dataset)):
        os.makedirs(os.path.join('output',args.model,args.dataset))


    root = os.path.join(args.data_dir, args.dataset)
    mean_path = os.path.join(root, 'means.npy')
    if not os.path.isfile(mean_path):
        save_mean(dataset(args.t, root=root, output_size=args.resolution, train=True), device, mean_path)

    mean, std = np.load(mean_path)
    train_transformer = transformer(output_size=args.resolution, mean=mean, std=std)
    valid_transformer = transformer(output_size=args.resolution,
                                    p_scale=0.0, p_flip=0.0, p_rotate=0.0,
                                    mean=mean, std=std)

    train_dataset = dataset(train=True, T=args.t, root=root, transformer=train_transformer,
                            output_size=args.resolution, sigma_center=21, sigma_label=2)
    valid_dataset = dataset(train=False, T=args.t, root=root, transformer=valid_transformer,
                            output_size=args.resolution, sigma_center=21, sigma_label=2)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size*num_devices, shuffle=True, **loader_args)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size*num_devices, shuffle=False, **loader_args)

    if args.block == 'residual':
        block = ResidualBlock
    elif args.block == 'inverted':
        block = InvertedResidualBlock
    else:
        block = ConvolutionalBlock

    criterion = MSESequenceLoss()

    if args.model == 'hourglass':
        model = PretrainRecurrentStackedHourglass(3, 64, n_joints + 1, device, block, T=args.t, depth=args.depth)
    elif args.model == 'lpm':
        model = LPM(3, 32, n_joints + 1, device, T=args.t)
    elif args.model == 'cpm':
        model = CPM(k=14)
    elif args.model == 'lpm_raw':
        model = LPM_raw(3, 32, n_joints + 1, device, T=args.t)


    if args.init_weight is not None:
        model.init_weights(args.model_dir,args.model,args.dataset,args.init_weight)

    model = torch.nn.DataParallel(model).cuda()

    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    summary = None
    if args.host is not None:
        cc = pycrayon.CrayonClient(hostname=args.host)
        if args.experiment in cc.get_experiment_names():
            summary = cc.open_experiment(args.experiment)
        else:
            summary = cc.create_experiment(args.experiment)

    best_acc = 0.
    start_epoch = 0

    if args.checkpoint_name is not None:
        checkpoint = load_checkpoint(args.model_dir,'{}/{}'.format(args.model,args.dataset), args.checkpoint_name, model, optimizer)
        start_epoch = checkpoint['epoch']
        if start_epoch>args.max_epochs:
            start_epoch = 0

    output_root = os.path.join(os.getcwd(),'output',args.model, args.dataset)
    log = log_creater(output_root)

    try:
        for epoch in range(start_epoch, args.max_epochs):
            log.info('\n[epoch ' + str(epoch) + ']')
            
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, logger,log,args.log_freq, device, args.pck_r,output_root,epoch, scheduler,
                                          args.clip, summary, args.debug)
            print('validating ....')
            valid_loss, valid_acc,eval_dic = validate(model, valid_loader,args.log_freq,output_root, criterion, device, args.pck_r)

            log.info('Epoch Valid Loss: {}, Epoch Valid Accuracy: {}'.format(valid_loss, valid_acc))
            print('Epoch Valid Loss: {}, Epoch Valid Accuracy: {}'.format(valid_loss, valid_acc))
            log.info('Head={}|Sholder={}|Elbow={}|Wrist={}|Hip={}|Knee={}|Ank={}|'.format(eval_dic[0],eval_dic[1],eval_dic[2],eval_dic[3],eval_dic[4],eval_dic[5],eval_dic[6]))
            logger.add_scalar('Epoch Train Loss', train_loss, epoch)
            logger.add_scalar('Epoch Valid Loss', valid_loss, epoch)
            logger.add_scalar('Epoch Train Accuracy', train_acc, epoch)
            logger.add_scalar('Epoch Valid Accuracy', valid_acc, epoch)

            if scheduler is not None:
                logger.add_scalar('Learning Rate', scheduler.get_lr()[0], epoch)

            is_best = valid_acc >= best_acc
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            is_best=is_best, checkpoint=args.model_dir,
                            dataset_name='{}/{}'.format(args.model,args.dataset),
                            log=log,
                            prefix=start_time_prefix)
            best_acc = valid_acc if is_best else best_acc

        logger.close()
    except KeyboardInterrupt:
        if args.host is not None:
            print('Saving Tensorboard data...')
            zip_path = os.path.join(args.model_dir, start_time_prefix + args.model)
            summary.to_zip(zip_path)
            cc.remove_experiment(args.experiment)
        logger.close()
        sys.exit(0)


if __name__ == '__main__':
    parser = configargparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('--config', is_config_file=True, help='config file path')

    # Architecture
    parser.add('--model', default='lpm', type=str, choices=['hourglass', 'lpm', 'coord_lpm','cpm','lpm_raw'], help='model type')
    parser.add('--t', default=5, type=int, help='length of input sequences (i.e. # frames)')
    parser.add('--depth', default=4, type=int, help='depth of each hourglass module')
    parser.add('--block', default='res', type=str, choices=['res', 'inv', 'conv'], help='type of hourglass blocks')

    # Training

    parser.add('--init_weight',default=None,type=str,help='init weight from cpm')#'cpm_latest.pth.tar'
    parser.add('--lr', default=1e-3, type=float, help='base learning rate')
    parser.add('--step_size', default=None, type=int, help='period of learning rate decay')#TODO
    parser.add('--gamma', default=1, type=float, help='multiplicative factor of learning rate decay')
    parser.add('--batch_size', default=4, type=int, help='training batch size')###
    parser.add('--weight_decay', default=0, type=float, help='l2 decay coefficient')
    parser.add('--max_epochs', default=100, type=int, help='maximum training epochs')
    parser.add('--resolution', default=256, type=int, help='model input image resolution')
    parser.add('--subset_size', default=None, type=int, help='size of training subset (for sanity overfitting)')
    parser.add('--clip', default=None, type=float, help='maximum norm of gradients')

    # Tensorboard
    parser.add('--log_dir', default='./logs', type=str, help='tensorboard log directory')
    parser.add('--experiment', default='Pose Estimation Training', type=str, help='name of Tensorboard experiment')
    parser.add('--host', default=None, type=str, help='Tensorboard host name')
    parser.add('--log_freq', default=800, type=int, help='logging frequency')

    # Checkpoints
    parser.add('--checkpoint_name', default=None, type=str, help='checkpoint file name in experiments/ to resume from')
    parser.add('--model_dir', default='experiments', type=str, help='directory to store/load checkpoints from')

    # Other
    parser.add('--data_dir', default='data', type=str, help='directory containing data')
    parser.add('--gpus', default=None, type=str, help='gpu ids to perform training on')
    parser.add('--pck_r', default=0.2, type=float, help='r coefficient for pck computation')
    parser.add('--dataset', default='PennAction', type=str, choices=['PennAction', 'MPII', 'LSP'], help='dataset to train on')
    parser.add('--debug', action='store_true', help='visualize model inputs and outputs')

    main(parser.parse_args())
