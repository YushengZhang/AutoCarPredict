from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset_loader import JsonDataset
from torch.optim import lr_scheduler

import data_manager
import models
from models.model import LstmModel
from optimizers import init_optim
from utils.iotools import save_checkpoint
from utils.avgmeter import AverageMeter
from utils.logger import Logger
from utils.torchtools import count_num_param
from losses import LabelLoss
import csv

parser = argparse.ArgumentParser(
    description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='AutoJson',
                    help="dataset direction")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py): rmsprop|sgd|amsgrad|adam")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=128, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  # 0.0001
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[15, 30, 45], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.25, type=float,  # default is 0.1
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# Architecture
parser.add_argument(
    '-a',
    '--arch',
    type=str,
    default='lstm',
    choices=list('lstm'))  # pick from choices
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument(
    '--load-weights',
    type=str,
    default='',  # checkpoint_latest.pth.tar, default is ''
    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument(
    '--save_step',
    type=int,
    default=1,
    help="save the model every save-step")
parser.add_argument(
    '--eval-step',
    type=int,
    default=5,
    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='./log/lstm_auto_v0.0')  # given in args
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu_devices', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    train_dataset = data_manager.init_dataset(
        root=args.root,
        name='json',
        phase='train'
    )
    valid_dataset = data_manager.init_dataset(
        root=args.root,
        name='json',
        phase='valid'
    )

    test_dataset = data_manager.init_dataset(
        root=args.root,
        name='json',
        phase='test'
    )

    test_mask = test_dataset.mask

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        JsonDataset(train_dataset),
        num_workers=4,
        batch_size=args.train_batch,
        pin_memory=pin_memory,
        drop_last=True
    )

    validloader = DataLoader(
        JsonDataset(valid_dataset),
        num_workers=4,
        batch_size=args.test_batch,
        pin_memory=pin_memory,
        drop_last=True
    )

    testloader = DataLoader(
        JsonDataset(test_dataset),
        num_workers=4,
        batch_size=args.test_batch,
        pin_memory=pin_memory,
        drop_last=True
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(
        name=args.arch,
        use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    model.init_weights()
    criterion_label = LabelLoss()

    optimizer = init_optim(
        args.optim,
        model.parameters(),
        args.lr,
        args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        print("Loading pretrained weights from '{}'".format(args.load_weights))
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items(
        ) if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}".format(start_epoch))

    # if use_gpu:
    # str_ids = args.gpu_devices.split(',')
    # gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         gpu_ids.append(id)
    # model = nn.DataParallel(model, gpu_ids)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.evaluate:
        print("Evaluate only")
        start_evaluate_time = time.time()
        test_thetas = evaluate(model, testloader, use_gpu, args.test_batch, test_mask)
        # test_thetas = evaluate(model, validloader, use_gpu, args.test_batch, test_mask)
        evaluate_time = time.time() - start_evaluate_time
        print('Evaluate: {} secs'.format(evaluate_time))
        with open("auto_sample.csv", "r") as csvfiler:
            with open("test_thetas.csv", "w") as csvfilew:
                reader = csv.reader(csvfiler)
                for item in reader:
                    if reader.line_num == 1:
                        writer = csv.writer(csvfilew)
                        writer.writerow(['test_id', 'result'])
                        continue
                    writer = csv.writer(csvfilew)
                    writer.writerow([item[0], str(test_thetas[reader.line_num - 2])])
        # writer.writerows(map(lambda x: [x], test_thetas))
        return

    start_time = time.time()
    train_time = 0
    best_label_loss = np.inf
    best_epoch = 0
    #
    # print("==> Test")
    # label_loss = test(model, validloader, criterion_label, use_gpu, args.test_batch)
    # print("test label loss RMES() is {}".format(label_loss))
    #
    # print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(
            epoch,
            model,
            criterion_label,
            optimizer,
            trainloader,
            use_gpu)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        # save model every epoch
        if (epoch + 1) % args.save_step == 0:
            print("==> Now save epoch {} \'s model".format(epoch + 1))
            # if use_gpu:
            #     state_dict = model.state_dict() #  module.
            # else:
            state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch
            }, False, osp.join(args.save_dir, 'checkpoint_latest.pth'))

        # test model every eval_step
        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or \
                (epoch + 1) == args.max_epoch:
            print("==> Test")
            label_loss = test(model, validloader, criterion_label, use_gpu, args.test_batch)
            is_best = label_loss < best_label_loss

            if is_best:
                best_label_loss = label_loss
                best_epoch = epoch + 1

            # if use_gpu:
            #     state_dict = model.state_dict()
            # else:
            state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'label_loss': label_loss,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth'))  # .pth.tar

    print(
        "==> Best Label Loss {:.3}, achieved at epoch {}".format(best_label_loss, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print(
        "Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(
            elapsed,
            train_time))


def train(
        epoch,
        model,
        criterion_label,
        optimizer,
        trainloader,
        use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    hidden01, hidden02, hidden03 = model.init_hidden(1)

    end = time.time()
    trainloader_len = len(trainloader)
    for batch_idx, (batch_data, labels) in enumerate(trainloader):
        # data_time.update(time.time() - end)

        batch_data = batch_data.transpose(0, 1)
        labels = labels.transpose(0, 1)

        if use_gpu:
            batch_data, labels = batch_data.cuda(), labels.cuda()

        # print("batch_data size is {}".format(batch_data.shape))  # [1,64,7]
        # print("label size is {}".format(labels.shape))  # [1,64]

        output_thetas, hidden1, hidden2, hidden3, l2_loss = model(batch_data, hidden01, hidden02, hidden03)

        theta_loss = criterion_label(output_thetas, labels)

        loss = theta_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hidden01, hidden02, hidden03 = hidden1.data, hidden2.data, hidden3.data

        batch_time.update(time.time() - end)

        losses.update(loss.item())

        if (batch_idx + 1) % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch + 1,
                    batch_idx + 1,
                    trainloader_len,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses))

        end = time.time()


def test(model, validloader, criterion_label, use_gpu, test_batch):
    batch_time = AverageMeter()
    model.eval()
    hidden01, hidden02, hidden03 = model.init_hidden(test_batch)
    test_loss_list = []

    with torch.no_grad():  # no bp
        for batch_idx, (batch_data, labels) in enumerate(validloader):

            if use_gpu:
                batch_data, labels = batch_data.cuda(), labels.cuda()

            end = time.time()

            output_thetas, hidden1, hidden2, hidden3 = model(batch_data, hidden01, hidden02, hidden03)
            hidden01, hidden02, hidden03 = hidden1, hidden2, hidden3
            theta_loss = criterion_label(output_thetas, labels)
            test_loss_list.append(theta_loss)
            batch_time.update(time.time() - end)

    squre = [i ** 2 for i in test_loss_list]
    loss = (sum(squre) / len(squre)) ** 0.5

    print(
        "==> BatchTime(s)/BatchSize(json): {:.3f}/{}".format(batch_time.avg, args.test_batch))
    print("Results ----------")
    print("label_loss: {:.3}".format(loss))
    print("------------------")

    return loss


def evaluate(model, testloader, use_gpu, test_batch, test_mask):
    batch_time = AverageMeter()
    model.eval()
    hidden01, hidden02, hidden03 = model.init_hidden(test_batch)
    test_theta = []
    theta_temp = []

    with torch.no_grad():  # no bp
        for batch_idx, batch_data in enumerate(testloader):

            if use_gpu:
                batch_data = batch_data.cuda()

            end = time.time()
            output_thetas, hidden1, hidden2, hidden3 = model(batch_data, hidden01, hidden02, hidden03)
            hidden01, hidden02, hidden03 = hidden1, hidden2, hidden3
            batch_time.update(time.time() - end)
            theta_temp.append(output_thetas.squeeze().item())

        temp_index = 0
        theta_len = len(theta_temp)

        for i, _ in enumerate(test_mask):
            if _:
                test_theta.append(theta_temp[temp_index])
                temp_index += 1
            else:
                # test_theta.append(0)
                if temp_index != theta_len:
                    temp1 = theta_temp[temp_index - 1]
                    temp2 = theta_temp[temp_index]
                    temp = (temp1 + temp2) / 2
                    test_theta.append(temp)
                else:
                    test_theta.append(0)
                continue

    return test_theta


if __name__ == '__main__':
    main()
