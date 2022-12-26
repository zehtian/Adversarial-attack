# -*- coding: utf-8 -*-
"""
@author: tzh666
@context: 成员推断攻击主函数
"""
import logging
logging.basicConfig(level=logging.INFO, filename="mylog.log", filemode='w', format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import torch
import os
import random
import numpy as np
import torch.nn.functional as F

from model_data_init import model_init, data_init, model_train, test
from Adv_nq2 import adv_attack2
from Adv_nq1 import adv_attack1
from Adv_ours import adv_attack

from Adv_hsja import hsja_initialize
from Adv_deepfool import deepfool_initialize
from Adv_cw import cw_initialize
from Adv_boundary import ba_initialize


class Arguments:
    def __init__(self):
        # 数据集名称 -> [mnist, cifar10, imagenet]
        self.data_name = 'cifar10'
        # 攻击模式
        self.target_attack = False
        # 攻击范数-> [l2, linf]
        self.constraint = 'l2'
        if self.constraint == 'l2':
            self.attack_norm = 2
        if self.constraint == 'linf':
            self.attack_norm = np.inf
        # 优化方式
        self.opt_mode = 'MOM'  # ['SGD', 'MOM2', 'ADAM']
        # 攻击的样本总数
        self.num_samples = 200
        # 是否记录目标模型测试结果
        self.train_with_test = True
        # 攻击时是否采用子空间
        self.subspace_attack = False
        # 设置子空间维度
        self.subspace_dim = 15

        ## GPU设置
        self.use_gpu = True
        self.cuda_state = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_cpu = torch.device('cpu')

        if self.data_name == 'mnist':
            # 模型训练参数设置
            self.train_batch_size = 64
            self.test_batch_size = 64
            self.train_epoch = 10
            self.local_lr = 0.01
            # 对抗攻击参数设置
            self.clip_min = -0.4242
            self.clip_max = 2.8088
            self.class_num = 10
            # 攻击迭代次数
            self.attack_iter = 30
            # 攻击查询次数
            self.num_evals_boundary = 5000
            # 攻击学习率
            self.attack_lr = 0.005

        elif self.data_name == 'cifar10':
            # 模型训练参数设置
            self.train_batch_size = 64
            self.test_batch_size = 64
            self.train_epoch = 10
            self.local_lr = 0.001
            # 对抗攻击参数设置
            self.clip_min = -1
            self.clip_max = 1
            self.class_num = 10
            # 攻击迭代次数
            self.attack_iter = 200
            # 攻击查询次数
            self.num_evals_boundary = 5000
            # 攻击学习率
            self.attack_lr = 0.1


def set_target(model, train_loader, FL_params):
    ## 找到每个类的对应目标样本(tgt_list)
    # 定义一个存放样本的数组，数组下标表示对应的类别
    tgt_list = []
    for tgt_label in range(FL_params.class_num):
        tgt_label = torch.tensor(tgt_label)
        # 定义标志位，一旦找到了一个就停止
        tgt_temp = 0
        for idb, (XX_tgt, YY_tgt) in enumerate(train_loader):
            if tgt_temp:
                break
            for i_yy in range(len(YY_tgt)):
                # 找到了对应的样本
                if YY_tgt[i_yy] == tgt_label:
                    # ***还需确保模型预测准确，对于目标模型没那么准的时候，很有必要！！
                    if tgt_label == torch.argmax(model(XX_tgt[i_yy].unsqueeze(0).to(FL_params.device))).to(
                                FL_params.device_cpu):
                        tgt_temp = 1
                        tgt_tgt = XX_tgt[i_yy].unsqueeze(0).to(FL_params.device)
                        # 添加样本
                        tgt_list.append(tgt_tgt)
                        break
    # 确保每个标签都有对应的初始点
    assert len(tgt_list) == FL_params.class_num

    return tgt_list


def get_target(model, xx_tgt, yy_tgt, tgt_list, data_num, FL_params):
    # 设置具体每个样本对应的target样本
    # Targeted情况
    if FL_params.target_attack:
        # 随机设置与原始标签不同的目标标签
        while True:
            adv_label = torch.tensor([random.randint(0, FL_params.class_num - 1)], device=FL_params.device)
            if adv_label != yy_tgt:
                break
        # 找到对应batch中的目标样本，用于生成初始对抗数据
        tgt_tgt = tgt_list[adv_label.item()]
        print('Targeted|| data_num:{}, yy_label:{}, adv_label:{}, distance:{}'.
              format(data_num, yy_tgt, adv_label, torch.norm(tgt_tgt - xx_tgt, FL_params.attack_norm)))
    # Untargeted情况
    else:
        init_size = 10000
        for _ in range(init_size):
            # 向原始数据点添加逐渐增加的噪声，直到模型错误分类为止
            mask = np.random.binomial(n=1, p=0.1, size=np.prod(xx_tgt.cpu().numpy().shape))
            mask = mask.reshape(xx_tgt.cpu().numpy().shape)
            random_ = np.random.RandomState().uniform(FL_params.clip_min, FL_params.clip_max, size=xx_tgt.cpu().numpy().shape).astype(xx_tgt.cpu().numpy().dtype)

            mask = torch.tensor(mask, device=FL_params.device)
            random_ = torch.tensor(random_, device=FL_params.device)
            random_img = random_ * mask + xx_tgt * (1 - mask)

            adv_label = torch.argmax(model(random_img), dim=1).to(FL_params.device)
            # 找到了满足要求的目标样本
            if adv_label != yy_tgt:
                tgt_tgt = random_img
                print('Untargeted|| data_num:{}, yy_label:{}, adv_label:{}, distance:{}'.
                      format(data_num, yy_tgt, adv_label, torch.norm(tgt_tgt - xx_tgt, FL_params.attack_norm)))
                break

        else:
            # 为保证代码不中断，创建失败的数据将变为Targeted形式
            # raise AttributeError("Failed to draw a random image that is adversarial, attack failed.")
            print("Failed to draw a random image that is adversarial, attack failed.")
            # 随机设置与原始标签不同的目标标签
            while True:
                adv_label = torch.tensor([random.randint(0, FL_params.class_num - 1)], device=FL_params.device)
                if adv_label != yy_tgt:
                    break
            # 找到对应batch中的目标样本，用于生成初始对抗数据
            tgt_tgt = tgt_list[adv_label.item()]

    return tgt_tgt, adv_label


def adv_MIA():
    # ************************数据、模型加载****************************/
    # 参数数据初始化
    FL_params = Arguments()
    # 加载数据集，数据存储器
    train_loader, test_loader = data_init(FL_params)

    # 模型训练，得到目标模型
    # 如果已经存在训练好的模型，则直接使用，否则进行模型训练
    model = None
    path = "./model/model_" + FL_params.data_name + "_" + str(FL_params.train_epoch) + "_" + str(FL_params.local_lr) + ".pth"
    # path = "./model/model_cifar10_lenet.pth"
    if not os.path.exists(path):
        model_train(FL_params, train_loader, test_loader)
    else:
        model = model_init(FL_params.data_name).to(FL_params.device)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # 目标模型测试
    test(model, train_loader, test_loader, FL_params)

    # *************************对抗攻击*****************************/
    # 保存bn层和dropout层的参数信息
    model.eval()
    # 累计攻击的数据个数
    data_num = 0
    # 设置好目标数据矩阵，Targeted时需要
    tgt_list = set_target(model, train_loader, FL_params)

    ## 遍历所有batch中的数据
    dist1 = []
    dist2 = []
    for idb, (XX_tgt, YY_tgt) in enumerate(train_loader):

        for idx in range(FL_params.train_batch_size):

            # 限制输入数据个数
            if data_num >= FL_params.num_samples:
                break

            ## 输入数据设定
            # XX YY代表一组数据，xx yy代表该组中的一个数据
            # cifar10: X_ [64, 3, 32, 32]->[1, 3, 32, 32]
            # cifar10: Y_ [64]->[1]
            # cifar10: model(X)_ [1, 10]->[10, ]
            xx_tgt = XX_tgt[idx].unsqueeze(0).to(FL_params.device)
            yy_tgt = YY_tgt[idx].unsqueeze(0).to(FL_params.device)
            # 确保模型预测正确
            if yy_tgt != torch.argmax(model(xx_tgt), -1):
                continue

            ## 目标标签设置
            tgt_tgt, adv_label = get_target(model, xx_tgt, yy_tgt, tgt_list, data_num, FL_params)

            ## 开始进行对抗攻击
            # model->目标模型 xx_tgt->目标数据输入 yy_tgt->目标数据标签, tgt_tgt->目标攻击初始对抗数据
            # 1.OUR Attack
            d_1, _ = adv_attack1(model, xx_tgt, yy_tgt, tgt_tgt, adv_label, FL_params)
            d_1, _ = adv_attack2(model, xx_tgt, yy_tgt, tgt_tgt, adv_label, FL_params)
            # 2.HSJA Attack
            # x_2, d_2 = hsja_initialize(model, xx_tgt, yy_tgt, tgt_tgt, adv_label, FL_params)
            # dist2.append(d_2)
            # 3.DEEPFOOL
            # x_3, d_3, _ = deepfool_initialize(model, xx_tgt, yy_tgt, adv_label, FL_params)
            # 4.CW
            # x_4, d_4, _ = cw_initialize(model, xx_tgt, yy_tgt, adv_label, FL_params)
            # 5.BA
            # x_5, d_5 = ba_initialize(model, xx_tgt, yy_tgt, tgt_tgt, adv_label, FL_params)
            # dist2.append(d_5)

            data_num += 1

    dist1 = torch.tensor(dist1)
    dist2 = torch.tensor(dist2)

    torch.set_printoptions(threshold=np.inf)
    print('d_1:', dist1)
    print('d_avg1:', sum(dist1))
    print('d_2:', dist2)
    print('d_avg2:', sum(dist2))


if __name__ == "__main__":

    adv_MIA()
