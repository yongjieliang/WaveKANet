import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import setproctitle
from dataloader.dataset import MedicalDataSets
import albumentations as A
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import logging
import sys
from utils.util import AverageMeter
from utils.metrics import calculate_binary_classification_metric
from tensorboardX import SummaryWriter
from network.WaveKANet import wavekanet


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()


parser.add_argument('--model', type=str, default="wavekanet",
                    choices=["wavekanet"], help='model')
parser.add_argument('--base_dir', type=str, default="Datasets/TCGA", help='dir')
parser.add_argument('--train_file_dir', type=str, default="TCGA4096_class_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="TCGA4096_class_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='maximum epoch number to train')


args = parser.parse_args()


def getDataloader():
    img_size = 256
    train_transform = Compose([
        RandomRotate90(),
        A.HorizontalFlip(),
        Resize(img_size, img_size),
        A.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        A.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train", transform=train_transform,
                               train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False,
                           num_workers=4)
    return trainloader, valloader


def get_model(args):
    if args.model == "wavekanet":
        model = wavekanet()
    else:
        model = None
        print("model err")
    return model.cuda()


def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model = get_model(args)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    iter_num = 0
    max_epoch = args.max_epoch
    max_iterations = len(trainloader) * max_epoch
    args.snapshot_path = f'WaveKANet/log/{args.train_file_dir.split("_")[0]}/{args.model}/{args.batch_size}bs_{max_epoch}eps_{base_lr}lr'
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    logging.basicConfig(filename=args.snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(args.snapshot_path + '/log')
    logging.info(f'{len(trainloader)} iterations per epoch')
    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {'tra_loss': AverageMeter(),
                      'tra_acc': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_acc': AverageMeter(),
                      'sensitivity': AverageMeter(),
                      'specificity': AverageMeter(),
                      'precision': AverageMeter(),
                      'f1_score': AverageMeter()
                      }
        preds = []  # predicted class
        gts = []  # ground truth
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            loss = criterion(outputs, label_batch)
            outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
            label_batch = label_batch.cpu().detach().numpy()
            preds.extend(outputs)
            gts.extend(label_batch)
            accuracy, sensitivity, specificity, precision, f1 = calculate_binary_classification_metric(preds, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['tra_loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['tra_acc'].update(accuracy, volume_batch.size(0))
            writer.add_scalar('train_info/train_loss', avg_meters['tra_loss'].avg, epoch_num)
            writer.add_scalar('train_info/train_acc', avg_meters['tra_acc'].avg, epoch_num)
        preds = []  # predicted class
        gts = []  # ground truth
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                outputs = model(input)
                loss = criterion(outputs, target)
                outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                preds.extend(outputs)
                gts.extend(target)
                accuracy, sensitivity, specificity, precision, f1 = calculate_binary_classification_metric(preds, gts)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_acc'].update(accuracy, input.size(0))
                avg_meters['sensitivity'].update(sensitivity, input.size(0))
                avg_meters['specificity'].update(specificity, input.size(0))
                avg_meters['precision'].update(precision, input.size(0))
                avg_meters['f1_score'].update(f1, input.size(0))
                writer.add_scalar('val_info/val_loss', avg_meters['val_loss'].avg, epoch_num)
                writer.add_scalar('val_info/val_acc', avg_meters['val_acc'].avg, epoch_num)
                writer.add_scalar('val_info/Sensitivity', avg_meters['sensitivity'].avg, epoch_num)
                writer.add_scalar('val_info/Specificity', avg_meters['specificity'].avg, epoch_num)
                writer.add_scalar('val_info/Precision', avg_meters['precision'].avg, epoch_num)
                writer.add_scalar('val_info/F1', avg_meters['f1_score'].avg, epoch_num)

        logging.info(
            f'epoch [%d/%d]  tra_loss : %.4f, tra_acc: %.4f '
            '- val_loss %.4f - val_acc %.4f - Sensitivity %.4f - Specificity %.4f - Precision %.4f - F1 %.4f'
            % (epoch_num + 1, max_epoch, avg_meters['tra_loss'].avg, avg_meters['tra_acc'].avg,
            avg_meters['val_loss'].avg, avg_meters['val_acc'].avg, avg_meters['sensitivity'].avg,
            avg_meters['specificity'].avg, avg_meters['precision'].avg, avg_meters['f1_score'].avg))

        if avg_meters['val_acc'].avg > best_acc:
            if not os.path.exists('WaveKANet/checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(), 'WaveKANet/checkpoint/{}_model_{}.pth'
                       .format(args.model, args.train_file_dir.split(".")[0]))
            best_acc = avg_meters['val_acc'].avg
            logging.info("=> saved best model")
    logging.info(f'best acc: {best_acc}')
    logging.info(f'================================================================================')
    return "Training Finished!"

if __name__ == "__main__":
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    seed_torch(3407)
    torch.cuda.set_device(4)
    train(args)
