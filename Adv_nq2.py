# -*- coding: utf-8 -*-
"""
@author: tzh666
@context: 进行黑盒对抗攻击
"""
import torch
import copy
import numpy as np
import logging
import torch.nn.functional as F


def loss_sin(model, adv_list, x_adv, x_ori, adv_label, FL_params):
    """
    * 将原始数据与对抗样本的向量 与 两轮对抗样本连线的向量 的正弦相似度的相反数 作为损失函数
    * 把向量转成1维，求相似度
    :param adv_list: 近两轮对抗样本构成的列表
    :param x_adv: 决策边界上的对抗样本
    :param x_ori: 原始数据
    :param FL_params: 主函数参数集合
    :return sin: 损失值 -> [0, 1] 从 1 减到 0
    """
    # 原始数据与对抗样本的向量
    x_sub = x_adv - x_ori
    x_sub = x_sub.view(-1)  # 对于结构为多维的数据集(MNIST、CIFAR...)，需要将其扁平化再进行相似度求值
    # 得单位向量
    x_sub = x_sub / torch.norm(x_sub, p=FL_params.attack_norm)

    # 做上一轮样本与这一轮样本的对称点
    adv_list_2 = 2 * adv_list[1] - adv_list[0]
    # 生成连线单位向量
    s1 = adv_list[0] - adv_list[1]
    s1 = s1 / torch.norm(s1, p=FL_params.attack_norm)
    print('y0:', torch.argmax(model(adv_list_2)))
    print('y1:', model(adv_list[1]))
    print('y2:', model(adv_list_2))

    # 二分法投影到决策边界上
    if torch.argmax(model(adv_list_2)) == adv_label:
        adv_list_2 = binary_search(model, x_ori, adv_list_2, adv_label, FL_params)
        print('y3:', model(adv_list_2))
        # 生成连线单位向量
        s2 = adv_list_2 - adv_list[1]
        s2 = s2 / torch.norm(s2, p=FL_params.attack_norm)
        # 做梯度估计
        grad_xadv = s1 + s2
        grad_xadv = grad_xadv.view(-1)
        grad_xadv = grad_xadv / torch.norm(grad_xadv, p=FL_params.attack_norm)

    else:
        adv_list_2 = binary_search(model, adv_list_2, adv_list[2], adv_label, FL_params)
        print('y3:', model(adv_list_2))
        # 生成连线单位向量
        s2 = adv_list_2 - adv_list[1]
        s2 = s2 / torch.norm(s2, p=FL_params.attack_norm)
        # 做梯度估计
        grad_xadv = s1 + s2
        grad_xadv = grad_xadv.view(-1)
        grad_xadv = - grad_xadv / torch.norm(grad_xadv, p=FL_params.attack_norm)

    # 求余弦相似度
    sin = torch.cosine_similarity(x_sub, grad_xadv, dim=-1)

    print('d0(x_i->x_i-1):', torch.norm(adv_list[0] - adv_list[1], p=FL_params.attack_norm))
    print('d1(x_i+1->x_i):', torch.norm(adv_list_2 - adv_list[1], p=FL_params.attack_norm))

    return sin


def loss_cos(model, x_adv, x_ori, adv_label, distance, FL_params):
    """
    * 将原始数据与对抗样本的向量 与 对抗样本梯度估计的向量 的相关性的相反数 作为损失函数
    * 把向量转成1维，求相似度
    :param model: 目标模型
    :param x_adv: 决策边界上的对抗样本
    :param x_ori: 原始数据
    :param adv_label: 对抗目标标签
    :param distance: 上一步的距离值
    :param FL_params: 主函数参数集合
    :return cos: 损失值 -> [-1, 0] 从 0 减到 -1
    """
    # 原始数据与对抗样本的向量
    x_sub = x_adv - x_ori
    x_sub = x_sub.view(-1)  # 对于结构为多维的数据集(MNIST、CIFAR...)，需要将其扁平化再进行相似度求值
    # 得单位向量
    x_sub = x_sub / torch.norm(x_sub, p=FL_params.attack_norm)

    # 对抗样本梯度估计的向量
    grad_xadv = gradient_compute_boundary(model, x_adv, adv_label, distance, FL_params)
    grad_xadv = grad_xadv.view(-1)
    grad_xadv = grad_xadv / torch.norm(grad_xadv, p=FL_params.attack_norm)

    # 求余弦相似度
    cos = -torch.cosine_similarity(x_sub, grad_xadv, dim=-1)

    return cos


def get_random_noise(sample, num_evals, FL_params):

    # 设置扰动大小 rv->[B, 3, 32, 32]
    # 第零维：扰动个数(数据个数) 第一二三维：每个数据的shape
    noise_shape = [num_evals] + list(sample.shape)

    # 加载子空间数据
    if FL_params.subspace_attack:
        # sub_basis.shape = [32*32, dim*dim]
        sub_basis = np.load('2d_dct_basis_{}.npy'.format(FL_params.subspace_dim)).astype(np.float32)
        sub_basis = torch.from_numpy(sub_basis).to(FL_params.device)
        # noise.shape = [dim*dim, 3*B]
        noise = torch.randn([sub_basis.shape[1], 3 * num_evals], dtype=torch.float32).to(FL_params.device)
        # sub_noise.shape = [3*B, 32*32]
        # noise = FL_params.noise
        sub_noise = torch.transpose(torch.mm(sub_basis, noise), 0, 1)
        # rv.shape = [B, 3, 32, 32]
        rv = sub_noise.view(noise_shape)
    else:
        rv = torch.randn(*noise_shape, device=FL_params.device)

    rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=(1, 2, 3), keepdim=True))

    return rv


def gradient_compute_boundary(model, sample, adv_label, distance, FL_params):
    """
    * 采用蒙特卡洛估计法，计算决策边界上数据点的梯度大小，得到结果应于决策边界法向量一致
    * 均在gpu上进行操作
    :param model: 目标模型
    :param sample: 决策边界上的数据点
    :param adv_label: 对抗目标标签
    :param distance: 上一个对抗样本与原始样本的距离
    :param FL_params: 主函数参数集合
    :return gradf: 该样本梯度值
    * 输出梯度的形状应与输入数据点相同
    """
    # sample.shape = torch.size([1, 3, 32, 32])->torch.size([3, 32, 32])
    sample = sample.squeeze(0)

    # 设置扰动的组数，即蒙特卡洛估计公式中的B
    num_evals = FL_params.num_evals_boundary
    # 设置步长大小，即蒙特卡洛估计公式中的delta
    delta = 1 / len(sample.shape) * distance
    # 初始距离过大时，delta=0.2/0.5是一个好的调试参数
    # delta = 0.2

    # 随机生成B个扰动
    rv = get_random_noise(sample, num_evals, FL_params)

    # 得到扰动后的数据  shape->[B, 3, 32, 32]
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, FL_params.clip_min, FL_params.clip_max, FL_params)
    rv = (perturbed - sample) / delta

    # 模型预测，即fai x* (x)
    # decisions.shape->[B, 1]
    decisions = decision_function(model, perturbed, adv_label, FL_params)

    # decision_shape->[B, 1]
    decision_shape = [len(decisions)] + [1] * len(sample.shape)
    # 将结果从0/1变成-1/+1，其中对抗成功为+1
    fval = 2 * decisions.reshape(decision_shape) - 1.0
    avg_fval = torch.mean(fval)
    # print("avg_val:", avg_fval)

    # 得到最后的梯度
    if torch.mean(fval) == torch.tensor(1.0):
        gradf = torch.mean(rv, dim=0)
    elif torch.mean(fval) == torch.tensor(-1.0):
        gradf = - torch.mean(rv, dim=0)
    else:
        fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, dim=0)

    # gradf.shape->torch.size([3, 32, 32])
    return gradf


def clip_image(image, clip_min, clip_max, FL_params):
    """
    * 图片正则化，对数据范围进行限制
    :param image: 输入数据
    :param clip_min: 范围的下界
    :param clip_max: 范围的上界
    :param FL_params: 主函数参数集合
    :return: 正则化后的图片
    """
    return torch.min(torch.max(
                torch.tensor(float(clip_min), device=FL_params.device), image),
                torch.tensor(float(clip_max), device=FL_params.device))


def binary_search(model, x_0, x_random, adv_label, FL_params, tol=1e-5):
    """
    * 使用二分法，将对抗样本投影到决策边界上
    :param model: 目标模型
    :param x_0: 原始样本
    :param x_random: 扰动后得到的对抗样本
    :param adv_label: 对抗目标标签
    :param FL_params: 主函数参数集合
    :param tol: 二分阈值
    :return adv: 投影到决策边界上的最终样本x0
    """
    adv = x_random
    cln = x_0
    i = 0

    while True:
        # 二分操作
        mid = (cln + adv) / 2.0

        if decision_function(model, mid, adv_label, FL_params):
            adv = mid
        else:
            cln = mid

        i += 1

        if torch.norm(adv - cln).cpu().numpy() < tol:
            break

    return adv


def decision_function(model, images, adv_label, FL_params, batch_size=64):
    """
    Decision function output 1 on the desired side of the boundary,
	0 otherwise
    :param model: 目标模型，在GPU上进行
    :param images: 扰动后的数据点
    :param adv_label: 对抗目标标签
    :param FL_params: 主函数参数集合
    :return: 该数据点是否具有对抗性（1->有 0->无） shape=[len(images), 1]
    """
    model.train(mode=False)

    # 若只有一个数据，则直接预测
    if len(images) == 1:
        with torch.no_grad():
            predict_label = torch.argmax(model(images), dim=1).reshape(len(images), 1)

    # 数据较多，分批预测，减少时间
    else:
        results_list = []

        # Run prediction with batch processing
        num_batch = int(np.ceil(len(images) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, images.shape[0]),
            )
            with torch.no_grad():
                output = model(images[begin:end])
            output = output.detach().cpu().numpy().astype(np.float32)
            results_list.append(output)

        results = np.vstack(results_list)
        predict = torch.from_numpy(results).to(FL_params.device)
        predict_label = torch.argmax(predict, dim=1).reshape(len(images), 1)

    # 对target_label进行变换
    target_label = torch.zeros((len(images), 1), device=FL_params.device)
    for i in range((len(images))):
        target_label[i, 0] = adv_label

    return predict_label == target_label


def adv_attack2(model, x_sample, y_sample, tgt_sample, adv_label, FL_params):
    """
    黑盒对抗攻击主函数
    * 1.在决策边界上随机生成一个对抗样本；
    * 2.采用蒙特卡洛估计决策边界上样本的梯度；
    * 3.使用SGD，进行对抗样本更新；
    * 4.采用二分法投影更新后的样本
    * 注意：model，x_sample，y_sample，tgt_sample 均在 FL_params.device 上

    :param model: 目标模型
    :param x_sample: 原始样本输入特征 -> [1, 3, 32, 32]
    :param y_sample: 原始样本标签，仅用于打印显示 -> [1]
    :param tgt_sample: 随机生成的对抗样本 -> [1, 3, 32, 32]
    :param adv_label: 对抗目标标签 -> [1]
    :param FL_params: 主函数参数集合
    :return distance_value_l1: 生成对抗样本与原始样本的 L1 范数距离值
    :return distance_value: 生成对抗样本与原始样本的 指定 范数距离值（初始为L2）
    :return distance_value_linf: 生成对抗样本与原始样本的 L∞ 范数距离值
    :return distance_direction: 生成对抗样本与原始样本的单位方向向量
    :return cos_: 生成对抗样本与原始样本的单位方向向量 和 该对抗样本垂直与决策边界的梯度估计值 的余弦相似度
    :return is_adv: 是否对抗成功（1->成功 0->失败）

    """
    # 在决策边界上随机生成一个对抗样本
    adv_init = binary_search(model, x_sample, tgt_sample, adv_label, FL_params)
    adv_update = copy.deepcopy(adv_init)
    ori_sample = x_sample

    # 定义一个数组，储存最近两轮的样本+上一轮更新的对抗样本
    adv_list = [adv_init, adv_init, adv_init]

    # 定义一个存放距离的张量，一个代表大小、一个代表方向
    distance_init = torch.norm(adv_init - ori_sample, p=FL_params.attack_norm)
    distance_value = distance_init

    # 标志位，是否对抗成功
    is_adv = 0

    ## 对抗样本更新主循环
    # ori_sample: 原始样本（目标样本）
    # adv_sample: 正在更新的样本
    # adv_update: 每轮迭代后位于决策边界上的点，初始值为第0轮的点
    # distance_value: 每轮迭代后决策边界上的点到原始点的距离，初始值为第0轮的距离
    for attack_epoch in range(FL_params.attack_iter):

        # 对抗样本初始化
        # adv_sample.shape = [1, 3, 32, 32]
        adv_sample = copy.deepcopy(adv_update)

        # 距离值初始化
        distance = copy.deepcopy(distance_value)

        # 损失函数梯度计算
        adv_sample.requires_grad = True
        # 第一轮采用MC-MC估计梯度，采用梯度作为参考向量
        # if not attack_epoch or torch.abs(torch.norm(adv_list[0] - adv_list[1], p=2) - FL_params.attack_lr).item() < 1e-4:
        if not attack_epoch:
            loss = loss_cos(model, adv_sample, ori_sample, adv_label, distance, FL_params)
            print("adv_ours: epoch:{}, adv_label:{}, loss:{}, distance:{}".
                  format(attack_epoch, torch.argmax(model(adv_sample)), loss, distance))
        # 从第二轮开始，采用对称性进行梯度估计
        else:
            loss = loss_sin(model, adv_list, adv_sample, ori_sample, adv_label, FL_params)

        loss.backward()
        grads = adv_sample.grad
        grads = grads / torch.norm(grads, p=FL_params.attack_norm)
        adv_sample.requires_grad = False
        # **** 得到了该点的沿着损失函数减少最快方向的梯度值 ****

        # 打印更新方向与上一次方向的夹角
        # if attack_epoch:
        #     print('cos(d1~grad):', torch.cosine_similarity((adv_list[0] - adv_list[1]).view(-1), -grads.view(-1), dim=-1).item())

        # 进行对抗样本的更新
        if FL_params.opt_mode == 'SGD':
            # 收敛时增大学习率，跳出局部
            # if torch.norm(adv_list[0] - adv_list[1], p=FL_params.attack_norm).item() < (FL_params.attack_lr + 1e-3):
            # if torch.norm(adv_list[0] - x_sample, p=FL_params.attack_norm).item() \
            #         < torch.norm(adv_list[1] - x_sample, p=FL_params.attack_norm).item():
            lr = 0.2

            adv_sample -= lr * grads

        if FL_params.opt_mode == 'MOM1':

            lr = 0.8 / (2 ** (attack_epoch // 5))
            if lr < 0.01:
                lr = 0.01
            if attack_epoch and not attack_epoch % 10:
                lr *= 10
            print('lr:', lr)

            beta = 0.7
            if not attack_epoch:
                momentum = torch.zeros(3, 32, 32)
            momentum = beta * momentum + (1 - beta) * grads
            adv_sample -= lr * momentum

        if FL_params.opt_mode == 'MOM2':
            lr = 0.2
            beta = 0.7
            if not attack_epoch:
                momentum = torch.zeros(3, 32, 32)
            momentum = beta * momentum + (1 - beta) * grads
            adv_sample -= lr * momentum

        if FL_params.opt_mode == 'ADAM':
            lr = 0.001
            beta1 = 0.9
            beta2 = 0.999
            if not attack_epoch:
                m = grads
                v = torch.zeros(3, 32, 32)

            m = beta1 * m + (1.0 - beta1) * grads
            v = beta2 * v + (1.0 - beta2) * grads ** 2
            # m = m / (1.0 - beta1 ** (attack_epoch + 1))
            # v = v / (1.0 - beta2 ** (attack_epoch + 1))
            adv_sample -= lr * m / (torch.sqrt(v) + 1e-8)

        adv_sample = clip_image(adv_sample, FL_params.clip_min, FL_params.clip_max, FL_params)
        adv_list[2] = copy.deepcopy(adv_sample)

        # 二分法，将得到的样本投影到决策边界上
        adv_sample = binary_search(model, ori_sample, adv_sample, adv_label, FL_params)
        # print("is_adv3?", torch.argmax(model(adv_sample)))
        # print("二分法后样本预测：", F.softmax(model(adv_sample), dim=-1))

        # 原始数据与对抗数据的距离值大小
        distance_value = torch.norm(adv_sample - ori_sample, p=FL_params.attack_norm)
        print("adv_ours: epoch:{}, adv_label:{}, loss:{}, distance:{}".
              format(attack_epoch, torch.argmax(model(adv_sample)), loss, distance))

        # 保存更新之后的对抗样本
        adv_update = copy.deepcopy(adv_sample)
        adv_list[0] = copy.deepcopy(adv_list[1])
        adv_list[1] = copy.deepcopy(adv_update)

    print("our_attack: ori_sample:{}, adv_label:{}, distance:{}".format(y_sample, adv_label, distance_value))

    return distance_value, is_adv
