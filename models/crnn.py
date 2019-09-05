import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()           #seq  batch hidden
        t_rec = recurrent.view(T * b, h)  # hidden 是最后神经元的数量.

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]

            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))#

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        '''
        最后的在这里

        :param input:
        :return:
        '''
        conv = self.cnn(input) #先跑cnn
        print("conv.size",conv.size())

        # #torch.Size([1, 512, 1, 26])   得到的结果.最后图片变成了1,26
        b, c, h, w = conv.size()   #batch ,channle height ,weiht
        assert h == 1, "the height of conv must be 1"
        #从这里可以看出来,如果输入的图片高度 只能去32或者16.
        #这就是这个算法最大缺陷,只能识别单行的文本,不能识别多行文本的图片.
        conv = conv.squeeze(2) # b *512 * width  #把高度干的!
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        return output
import torch
'''
下面逐个层打印.看网络里面的shape ,把网络里面的设置贴到nn.什么就可.
'''

CRNN( 32  , 1  , 37  ,  256).forward(torch.randn(1,1,32,37))
a=torch.randn(1,1,32,100)# 图片32*100

#输入一个神经元,64个出的神经元. 因为stride 是1. 所以输出的shape不变还是32*100
b=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(a)   #torch.Size([1, 64, 32, 37])
b=nn.LeakyReLU(0.2, inplace=True)(b)#激活函数不改变shape
#stride=2所以行列都缩小到一半. 变成1,64,16,50
b=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(b)
# 1,128 ,16,50
b=nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(b)






b=CRNN( 32  , 1  , 37  ,  256).cnn(a) #torch.Size([1, 512, 1, 26])
b=        b.squeeze(2).permute(2,0,1) # b *512 * width  #把高度干的!       #26 1 512
b=nn.LSTM(512, 256, bidirectional=True)(b)   #输出还是26 1 512




b=CRNN( 32  , 1  , 37  ,  256).forward(a)   #torch.Size([26, 1, 37])

