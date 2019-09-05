import numpy as np
import sys, os
import time


#import pudb;pu.db
import cv2
sys.path.append(os.getcwd())
# crnn packages
import torch
from torch.autograd import Variable
import utils
import models.crnn as crnn
import alphabets
import params
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='test_images/test2.png', help='the path to your images')
opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'trained_models/crnn_Rec_done_1.pth'
crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
alphabet = str1
nclass = len(alphabet)+1
# print(nclass)   #6736   总共分了6735个汉子.
#nclass是词典+1,最后一个表示识别不出来.

# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ### ratio
    ### 280是中文训练集中图片的宽度，160是将原始图片缩小后的图片宽度
    w_now = int(image.shape[1] / (280 * 1.0 / params.imgW))
    h, w = image.shape
    image = cv2.resize(image, (0,0), fx=w_now/w, fy=params.imgH/h, interpolation=cv2.INTER_CUBIC)
    image = (np.reshape(image, (params.imgH, w_now, 1))).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(params.mean).div_(params.std)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    '''
    主要是针对model 在训练时和评价时不同的 Batch Normalization  和  Dropout 方法模式。

 

https://blog.csdn.net/jinxin521125/article/details/78435899

eval（）时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。

不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大。

 

https://zhuanlan.zhihu.com/p/26893755

model.eval()，让model变成测试模式，对dropout和batch normalization的操作在训练和测试的时候是不一样的
    '''
    preds = model(image)
    #preds: 26,1,37
    _, preds = preds.max(2)
    #preds: 坐标. 所以就得到了预测后的分类结果.
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':
    '''
    注意代码中是否使用cuda的书写
    '''
    tmp=    torch.cuda.is_available()
    print(tmp)
    map_location=torch.device('cuda')
	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    #注意要加上下面一句话来兼容cpu和gpu

    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path, map_location=map_location))

    started = time.time()
    ## read an image
    '''
    https://oldpan.me/archives/pytorch-conmon-problem-in-training 看这个试试
    '''
    image = cv2.imread(opt.images_path)
    if torch.cuda.is_available():
       model = model.cuda()

    print('tes!!!!!!!!!')
    tmp=[i.is_cuda for i in list(model.parameters())]
    print(tmp)
    print('tes!!!!!!!!!')
    crnn_recognition(image, model)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))

