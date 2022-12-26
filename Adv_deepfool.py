# -*- coding: utf-8 -*-
"""
@author: tzh666
@context: 生成初始对抗样本x0(使用DEEPFOOL算法)
"""
from AdvBox.adversarialbox.adversary import Adversary
from AdvBox.adversarialbox.attacks.deepfool import DeepFoolAttack
from AdvBox.adversarialbox.models.pytorch import PytorchModel

import torch
import numpy as np

def deepfool_initialize(model, x_sample, y_sample, adv_label, FL_params):
    """
    * 采用deepfool攻击，在决策边界处初始化一个对抗样本
    :param model: 目标模型
    :param x_sample: 目标数据的输入值
    :param y_sample: 目标数据的输出标签值
    :param adv_label: 对抗标签值
    :param FL_params: 参数信息
    :return: 生成的对抗样本及距离信息，所有信息均在cpu上
    """
    model = model.to(FL_params.device)
    model = model.eval()
    XX_target = x_sample.to(FL_params.device_cpu)
    YY_target = None

    M_tgt = PytorchModel(model, None, bounds=(FL_params.clip_min, FL_params.clip_max), channel_axis=1)

    deepfool_attacker = DeepFoolAttack(M_tgt)

    attacker_config = {"iterations": 1000, "overshoot": 0.02}

    adversary = Adversary(XX_target, YY_target)
    adversary.set_target(is_targeted_attack=True, target_label=adv_label.item())

    adversary = deepfool_attacker(adversary, **attacker_config)

    # 定义一个标志位，判断数据是否预测成功
    suc = 0
    if adversary.is_successful():
        suc = 1
        adv_s = adversary.adversarial_example[0]
        adv_s = torch.from_numpy(adv_s).unsqueeze(0)

        d = torch.norm(XX_target - adv_s, p=FL_params.attack_norm)

    else:
        adv_s = adversary.bad_adversarial_example[0]
        adv_s = torch.from_numpy(adv_s).unsqueeze(0)

        d = torch.norm(XX_target - adv_s, p=FL_params.attack_norm)

    print("deepfool_attack: ori_sample={}, adv_label={}, distance={}".
          format(y_sample, adversary.adversarial_label, d))

    return adv_s, d, suc

