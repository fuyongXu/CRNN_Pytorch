from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils_1
from datatset import dataset
import models.crnn as crnn
import re
import params


# argparse是python标准库里面用来处理命令行参数的库 使用步骤：（1）import argparse    首先导入模块（2）parser = argparse.ArgumentParser（）    创建一个解析对象
# （3）parser.add_argument()    向该对象中添加你要关注的命令行参数和选项 （4）parser.parse_args()    进行解析

def init_args():
    args = argparse.ArgumentParser()

    args.add_argument('--trainroot', help='path to dataset', default='./to_lmdb/train')
    args.add_argument('--valroot', help='path to dataset', default='./to_lmdb/train')
    # 'store_true'和'store_false' - 它们是'store_const' 的特殊情形，分别用于保存值True和False。另外，它们分别会创建默认值False 和True。例如：
    args.add_argument('--cuda', action='store_true', help='enables cuda', default=False)

    return args.parse_args()


def weight_init(m):
    # 实例调用__class__属性时会指向该实例对应的类
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def trainBatch(crnn, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    # 将cpu_images copy到image
    utils_1.loadData(image, cpu_images)
    # 得到text和length
    t, l = converter.encode(cpu_texts)
    utils_1.loadData(text, t)
    utils_1.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size

    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def training(crnn, train_loader, criterion, optimizer):
    for total_steps in range(params.niter):
        train_iter = iter(train_loader)
        i = 0
        print(len(train_loader))
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss:%f' % (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
            if i % params.valInterval == 0:
                val(crnn, test_dataset, criterion)

            if (total_steps + 1) & params.saveInterval == 0:
                torch.save(crnn.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.experiment, total_steps, i))


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers)
    )
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils_1.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils_1.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils_1.loadData(text, t)
        utils_1.loadData(length, l)
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        list_1 = []
        for i in cpu_texts:
            list_1.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_preds, pred, gt in zip(raw_preds, sim_preds, list_1):
        print('%-20s=>%-20s,gt:%-20s' % (raw_preds, pred, gt))

    print(n_correct)
    print(max_iter * params.batchSize)
    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Test loss:%f,accuracy:%f' % (loss_avg.val(), accuracy))


if __name__ == '__main__':
    args = init_args()
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

    # 模型存储路径
    if not os.path.exists('./expr'):
        os.mkdir('./expr')

    # 读取训练数据
    train_dataset = dataset.lmdbDataset(root=args.trainroot)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None

    # https://www.cnblogs.com/ranjiewen/p/10128046.html
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio)
    )

    # 读取测试数据，训练和测试数据的大小都得设置为32*160
    test_dataset = dataset.lmdbDataset(root=args.valroot, transform=dataset.resizeNormalize((160, 32)))

    nclass = len(params.alphabet) + 1
    # 只有一个通道
    nc = 1
    converter = utils_1.strLabelConverter(params.alphabet)
    # criterion = CTCLoss()
    criterion = torch.nn.CTCLoss()

    # cnn和rnn
    image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5)
    length = torch.IntTensor(params.batchSize)

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    if args.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weight_init)
    if params.crnn != '':
        print('loading pretrained model from %s' % params.crnn)
        crnn.load_state_dict(torch.load(params.crnn))

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils_1.averager()

    # 设置优化器
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    training(crnn, train_loader, criterion, optimizer)
