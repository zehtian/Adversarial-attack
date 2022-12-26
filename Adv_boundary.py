import torch
import numpy as np
from ToolBox.art.estimators.classification import PyTorchClassifier
from ToolBox.art.attacks.evasion import BoundaryAttack
from ToolBox.art.utils import to_categorical

def ba_initialize(model, x_sample, y_sample, tgt_sample, adv_label, FL_params):
    """
    * 采用hsja攻击，在决策边界处初始化一个对抗样本
    :param model: 目标模型
    :param x_sample: 目标数据的输入值
    :param y_sample: 目标数据的输出标签值
    :param tgt_sample: 初始化的对抗样本
    :param FL_params: 参数信息
    :return: 生成的对抗样本及距离信息，所有信息均在cpu上
    """
    # x_sample.shape -> torch.Size([1, 3, 32, 32])
    # 在cpu上运行ba
    # model = model.to('cpu')
    model = model.eval()
    x_sample = x_sample.to('cpu')
    y_sample = y_sample.to('cpu')
    tgt_sample = tgt_sample.to('cpu')
    adv_label = adv_label.to('cpu')

    # 判断攻击方式
    if not FL_params.target_attack:
        x_adv = None
        y = None
    else:
        x_adv = np.array(tgt_sample)
        y = to_categorical([adv_label], FL_params.class_num)

    M_tgt = PyTorchClassifier(model=model, loss=torch.nn.modules.loss.CrossEntropyLoss, input_shape=(1, 3, 32, 32), clip_values=(FL_params.clip_min, FL_params.clip_max),  nb_classes=FL_params.class_num)

    attack = BoundaryAttack(estimator=M_tgt, batch_size=FL_params.train_batch_size, targeted=FL_params.target_attack, max_iter=500, num_trial=20)

    x_adv = attack.generate(x=np.array(x_sample), y=y, x_adv_init=x_adv, resume=True)

    x_sample = torch.tensor(x_sample, device=FL_params.device)
    x_adv = torch.tensor(x_adv, device=FL_params.device)
    d = torch.norm(x_adv - x_sample, p=FL_params.attack_norm)
    adv_label = model(x_adv).argmax(-1)
    print("boundary attack: ori_label:{}, adv_label:{}, distance:{}".format(y_sample, adv_label, d))

    return x_adv, d
