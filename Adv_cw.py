# -*- coding: utf-8 -*-
"""
@author: tzh666
@context: 生成初始对抗样本x0(使用CW算法)
"""
import logging
import torch
import numpy as np
from AdvBox.adversarialbox.adversary import Adversary
from AdvBox.adversarialbox.attacks.cw2_pytorch import CW_L2
from AdvBox.adversarialbox.models.pytorch import PytorchModel

def cw_initialize(model, x_sample, y_sample, adv_label, FL_params):
    """
    * 采用cw攻击，在决策边界处初始化一个对抗样本
    :param model: 目标模型
    :param x_sample: 目标数据的输入值
    :param y_sample: 目标数据的输出标签值
    :param adv_label: 对抗标签值
    :param FL_params: 参数信息
    :return: 生成的对抗样本及距离信息，所有信息均在cpu上
    """
    # x_sample.shape -> torch.Size([1, 3, 32, 32])

    model = model.to(FL_params.device)
    model = model.eval()
    XX_target = x_sample.to(FL_params.device_cpu)

    YY_target = None

    M_tgt = PytorchModel(model, None, bounds=(FL_params.clip_min, FL_params.clip_max), channel_axis=1)

    cw_attacker = CW_L2(M_tgt)

    # 需要进行调参：initial_const（0.001、0.1、1、10中选）、max_iterations（好像没啥用）、binary_search_steps（根据情况，找个小的值）
    # 隐藏调参值：learning_rate（初始为0.01，也没啥用）、k（初始为40）
    attacker_config = {"num_labels": FL_params.class_num, "max_iterations": 1000, "binary_search_steps": 10, "initial_const": 1.0, "k": 0}

    adversary = Adversary(XX_target, YY_target)
    adversary.set_target(is_targeted_attack=True, target_label=adv_label.item())

    adversary = cw_attacker(adversary, **attacker_config)

    # 定义一个标志位，判断数据是否预测成功
    suc = 0
    if adversary.is_successful():
        suc = 1
        adv_s = adversary.adversarial_example[0]
        adv_s = torch.from_numpy(adv_s)

        d = torch.norm(XX_target - adv_s, p=FL_params.attack_norm)

    else:
        adv_s = adversary.bad_adversarial_example[0]
        adv_s = torch.from_numpy(adv_s)

        d = torch.norm(XX_target - adv_s, p=FL_params.attack_norm)

    print("cw_attack: ori_sample={}, adv_label={}, distance={}".
          format(y_sample, adversary.adversarial_label, d))

    return adv_s, d, suc
