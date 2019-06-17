import torch.nn as nn

class BidirectionlLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionlLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        #输出层的全连接操作
        self.embedding = nn.Linear(nHidden * 2, nOut)      #嵌入向量

    def forward(self, input):
        # LSTM前向运算 out,(h,c) = self.lstm(input,h,c) :https://zhuanlan.zhihu.com/p/39191116
        # h:保存着batch中每个元素的初始化隐状态的Tensor
        # c:保存着batch中每个元素的初始化细胞状态的Tensor
        recurrent, _ = self.rnn(input)
        #每个时间步骤上LSTM单元都会有一个输出，batch_size个样本并行计算
        # (每个样本/序列长度一致)  out (sequence_length,batch_size,hidden_size)
        # 把LSTM的输出结果变更为(sequence_length*batch_size, hidden_size)的维度
        #out = out.reshape(out.size(0)*out.size(1),out.size(2))
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  #[T*b,nout]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    #                   32   1    37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]        #卷积核的大小
        ps = [1, 1, 1, 1, 1, 1, 0]        #padding的大小
        ss = [1, 1, 1, 1, 1, 1, 1]        #stride的大小
        nm = [64, 128, 256, 256, 512, 512, 512]       #特征图的数量

        cnn = nn.Sequential()

        #将卷积和Relu整合在一起
        def convRelu(i, batchNormalization=False):
            #第0层没有输入特征数的数量
            nIn = nc if i == 0 else nm[i-1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))  #当x>0:x x<0:0.2*x
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))    #64*16*64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))    #128*8*32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  #256*4*16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512*2*16
        convRelu(6, True)               #512*1*16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionlLSTM(512, nh, nh),
            BidirectionlLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        #conv features
        conv = self.cnn(input)

        #print(conv.size()) batch_size*512*1*width
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # b*512*width
        conv = conv.permute(2, 0, 1)  # [w,b,c]
        # print(conv.size()) #width batch_size channel
        # rnn features
        output = self.rnn(conv)
        return output
