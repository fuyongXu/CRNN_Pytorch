import cv2
from math import *
import math
import numpy as np
import os
import glob

'''旋转图像并剪裁'''


def rotate(
        img,  # 图片
        pt1, pt2, pt3, pt4,  # 四点坐标
        NewimageName  # 输出图片路径
):
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    #    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    if (withRect != 0):
        angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度

        if pt4[1] < pt1[1]:
            angle = -angle

        height = img.shape[0]  # 原始图像高度
        width = img.shape[1]  # 原始图像宽度
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))

        # 旋转后图像的四点坐标
        [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
        [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
        [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

        # 处理反转的情况
        if pt2[1] > pt4[1]:
            pt2[1], pt4[1] = pt4[1], pt2[1]
        if pt1[0] > pt3[0]:
            pt1[0], pt3[0] = pt3[0], pt1[0]

        imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
        cv2.imwrite(NewimageName, imgOut)  #保存得到的旋转后的矩形框
        return imgRotation  # rotated image


# 读取四点坐标
def ReadTxt(directory, last):
    global Newpathofimage, Newpathoftxt, allpic, nowimage, nowtxt, nowline, invalidimg
    SetofimageName = glob.glob('*.jpg')  # 获取当前路径下所有jpg格式文件名到list中
    Numofimage = len(SetofimageName)
    for j in range(Numofimage):
        print('处理图片:' + str(j))
        imageTxt = directory + SetofimageName[j][:-4] + last  # txt路径
        imageName = SetofimageName[j]
        nowimage = imageName
        nowtxt = imageTxt
        nowline = 0
        imgSrc = cv2.imread(imageName)
        if (imgSrc is None):
            invalidimg.append(nowimage)
        else:
            F = open(imageTxt, 'rb')  # 以二进制模式打开目标txt文件
            lines = F.readlines()  # 逐行读入内容
            length = len(lines)
            s = 0  # 计算图片编号，对应文本描述
            for i in range(length):
                lines[i] = str(lines[i], encoding="utf-8")  # 从bytes转为str格式
                des = lines[i].split(',')[-1:]
                nowline = i
                if ((des != ['###\n']) and (des != ['###'])):
                    s = s + 1
                    allpic += 1
                    # 保存新图片/txt格式为"原名字+编号+.jpg/.txt"
                    NewimageName = Newpathofimage + imageName[:-3] + str(s) + '.jpg'
                    NewtxtName = Newpathoftxt + imageName[:-3] + str(s) + '.txt'
                    # 写入新TXT文件
                    if (s == length):
                        des = str(des)[2:-2]
                    else:
                        des = str(des)[2:-4]
                    file = open(NewtxtName, 'w',encoding='utf-8')  # 打开or创建一个新的txt文件
                    file.write(des)  # 写入内容信息
                    file.close()
                    # str转float
                    pt1 = list(map(float, lines[i].split(',')[:2]))
                    pt2 = list(map(float, lines[i].split(',')[2:4]))
                    pt3 = list(map(float, lines[i].split(',')[4:6]))
                    pt4 = list(map(float, lines[i].split(',')[6:8]))
                    # float转int
                    pt1 = list(map(int, pt1))
                    pt2 = list(map(int, pt2))
                    pt4 = list(map(int, pt4))
                    pt3 = list(map(int, pt3))
                    rotate(imgSrc, pt1, pt2, pt3, pt4, NewimageName)


if __name__ == "__main__":
    Newpathofimage = 'D:\\study\\天池对抗算法大赛\\天池文本识别\\mtwi_2018_train\\ewimage\\'
    Newpathoftxt = 'D:/study/天池对抗算法大赛/天池文本识别/mtwi_2018_train/newtxt/'
    allpic = 0
    nowimage = ''
    nowtxt = ''
    nowline = 0
    invalidimg = []
    os.chdir('D:/study/天池对抗算法大赛/天池文本识别/mtwi_2018_train/image_train')  # 修改默认路径
    retval = os.getcwd()  # 获取当前路径
    directory = 'D:/study/天池对抗算法大赛/天池文本识别/mtwi_2018_train/txt_train/'  # TXT文件路径
    last = '.txt'
    ReadTxt(directory, last)