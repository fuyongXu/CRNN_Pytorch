#!/usr/bin/python
#encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np

class lmdbDataset(Dataset):
    def __init__(self,root=None,transform = None,target_transform = None):
        #https://lmdb.readthedocs.io/en/release/#lmdb.lmdb.open
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock = False,
            readahead=False,
            meminit=False
        )
        if not self.env:
            print('connot creat lmdb from %s'%(root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            str = 'num-samples'.encode('utf-8')
            nSamples = int(txn.get(str))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples+1

    def __getitem__(self, index):
        assert index<=len(self),'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'%index
            imgbuf = txn.get(img_key.encode('utf-8'))

            #BytesIO专门读字符串
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d'%index)
                return self[index+1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d'%index
            label = txn.get(label_key.encode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img,label)

class resizeNormalize(object):
    def __init__(self,size,interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size,self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

    #%相当于mod，也就是计算除法的余数，比如5%3就得到2。
    #" / "  表示浮点数除法，返回浮点结果;
    #" // " 表示整数除法,返回不大于结果的一个最大的整数
    #__iter__https://blog.csdn.net/xrinosvip/article/details/86516276
class randomSequentialSampler(sampler.Sampler):
    def __init__(self,data_source,batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self)//self.batch_size
        tail = len(self)%self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        #每一次batch迭代都随机选择一个batch的样本
        for i in range(n_batch):
            #在0-len(self)-self.batch_size随机选择一个开始选择样本的起始点，只要留够一个batch
            random_start = random.randint(0,len(self)-self.batch_size)
            batch_index = random_start + torch.range(0,self.batch_size-1)
            #这里的index得根据i递增位置
            index[i*self.batch_size:(i+1)*self.batch_size] = batch_index
        #处理tail
        if tail:
            random_start = random.randint(0,len(self)-self.batch_size)
            tail_index = random_start+torch.range(0,tail-1)
            #只去剩下的部分
            index[(i+1)*self.batch_size:] = tail_index

        #返回一个迭代器
        return iter(index)

    def __len__(self):
        return self.num_samples

#对齐校正
class alignCollate(object):
    def __init__(self,imgH=32,imgW=256,keep_ratio=False,min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        #如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        images,labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w,h = image.size()
                ratios.append(w/float(h))
            #排序之后取最大值
            ratios.sort()
            max_ratio = ratios[-1]
            #按照元素的方式返回输入的下限。标量x的底部是最大的整数i，使得i <= x。
            imgW = int(np.floor(max_ratio*imgH))
            imgW = max(imgH*self.min_ratio,imgW)

        transforms = resizeNormalize((imgW,imgH))
        images = [transforms(image)for image in images]
        images = torch.cat([t.unsqueeze(0)for t in images],0)

        return images,labels