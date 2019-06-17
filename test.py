import numpy as np
import sys, os
import time

# os.getcwd()返回当前工作目录
sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
import utils_1
from datatset import dataset
from PIL import Image
import models.crnn as crnn
import alphabets

str1 = alphabets.alphabet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='D:/Project/CRNN_Pytorch/test_images/line_5.jpg',
                    help='the path to your images')
opt = parser.parse_args()

# crnn模型
crnn_model_path = 'D:/Project/CRNN_Pytorch/trained_models/acc97.pth'
alphabet = str1
nclass = len(alphabet) + 1


# crnn文本信息识别
def crnn_recognition(cropped_image, model):
    converter = utils_1.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    # view操作不会开辟新的内存空间来存放处理之后的数据，实际上新数据与原始数据共享同一块内存，view(1,6)则为一维6个元素
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    # 使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形。
    # 一种是维度变换后tensor在内存中不再是连续存储的，而view操作要求连续存储，所以需要contiguous
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    print(preds)
    print(preds_size)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('result:{0}'.format(sim_pred))


if __name__ == '__main__':
    # 准备网络
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from{0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    started = time.time()
    image = Image.open(opt.images_path)
    crnn_recognition(image, model)
    finished = time.time()
    print('elapsed time:{0}'.format(finished - started))
