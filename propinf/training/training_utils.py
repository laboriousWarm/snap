import time
import torch
import numpy as np
import pandas as pd

from torch import nn
from sklearn import metrics
import torch.optim as optim

# 就是把pandas转为tensor
def dataframe_to_torch_dataset(dataframe, using_ce_loss=False, class_label=None):
    """Convert a one-hot pandas dataframe to a PyTorch Dataset of Tensor objects"""
    # 拷贝副本
    new = dataframe.copy()
    # 获取标签列列名
    if class_label:
        label = class_label
    else:
        label = list(new.columns)[-1]
        # print(f"Inferred that class label is '{label}' while creating dataloader")
    # 先转换标签列
    # 不用再转换为dataframe，因为tensor接受series作为参数
    # labels = torch.Tensor(new[label])
    # 可以不要.values,这是用来提取pandas数据结构中底层的numpy数组，不过惯例是需要的
    labels = torch.Tensor(pd.DataFrame(new[label]).values)
    # print(labels)
    # 删除，不过之前不都是drop吗
    del new[label]

    data = torch.Tensor(new.values)

    # 如果用的交叉熵损失函数
    if using_ce_loss:
        # Fixes tensor dimension and float -> int if using cross entropy loss
        # squeeze(): 移除 labels 张量中所有单维度条目（即维度为 1 的维度）。
        # 例如，如果 labels 的形状是 [batch_size, 1]，squeeze() 会将其转换为 [batch_size]
        # 使用type(torch.LongTensor)是因为CrossEntropyLoss期望标签是整数类型
        # 函数内部会使用这些整数作为索引来从权重矩阵（即分类层的输出）中选择概率值
        # 有现版本的方法.type(torch.LongTensor)替换为.long()或者.to(dtype=torch.long)
        return torch.utils.data.TensorDataset(
            data, labels.squeeze().type(torch.LongTensor)
        )
    else:
        # torch.utils.data.TensorDataset(data, labels): 使用处理过的 data 和 labels
        # 创建一个 PyTorch 的 TensorDataset。这个数据集可以用于 PyTorch 的 DataLoader
        return torch.utils.data.TensorDataset(data, labels)


def dataset_to_dataloader(
    dataset, batch_size=256, num_workers=4, shuffle=True, persistent_workers=False
):
    """Wrap PyTorch dataset in a Dataloader (to allow batch computations)"""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
    )
    return loader


def dataframe_to_dataloader(
    dataframe,
    batch_size=256,
    num_workers=4,
    shuffle=True,
    persistent_workers=False,
    using_ce_loss=False,
    class_label=None,
):
    """Convert a pandas dataframe to a PyTorch Dataloader"""
    # 转pandas为tensor
    dataset = dataframe_to_torch_dataset(
        dataframe, using_ce_loss=using_ce_loss, class_label=class_label
    )
    # 转为dataloader，默认打乱重排。
    return dataset_to_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
    )


# 这跟也没用，就是作者在跑结果的时候，打印信息的，但是提供给我们的就只需要执行中毒准确率就好。
def get_metrics(y_true, y_pred):
    """Takes in a test dataloader + a trained model and returns a numpy array of predictions
    ...
    Parameters
    ----------
        y_true: Ground Truth Predictions

        y_pred: Model Predictions

    ...
    Returns
    -------
        Accuracy, Precision, Recall, F1 score
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    # 打印precision，recall，f1
    # precision = metrics.precision_score(y_true, y_pred)
    # recall = metrics.recall_score(y_true, y_pred)
    # f1 = metrics.f1_score(y_true, y_pred)

    # return acc, precision, recall, f1
    return acc

# 这个函数都没有调用，应该是用于测试
def get_prediction(test_loader, model, one_hot=False, ground_truth=False, device="cpu"):
    """Takes in a test dataloader + a trained model and returns a numpy array of predictions
    ...
    Parameters
    ----------
        test_loader : PyTorch Dataloader
            The Pytorch Dataloader for Dtest

        model : torch.nn.Module (PyTorch Neural Network Model)
            A trained model to be queried on the test data

        one_hot: bool
            If true, returns predictions in one-hot format
            else, returns predictions in integer format

        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:0

    ...
    Returns
    -------
        attackdata_arr : np.array
            Numpy array of predictions
    """
    # 转换为指定设备
    model = model.to(device)
    # 一个是存放预测结果，一个存放真实标签
    y_pred_torch = torch.Tensor([])
    y_true_torch = torch.Tensor([])

    for d, l in test_loader:
        # 测试数据
        d = d.to(device)
        # squeeze()函数用于删除输入张量中所有大小为1的维度
        # 测试标签
        l = l.squeeze()
        # 评估模式
        model.eval()
        # with torch.no_grad():
        # 用的sigmoid激活函数。之前都是softmax
        # 预测的输出
        # 这里因为模型的类别是2，所以相当于单独对每个类别应用sigmoid很奇怪啊
        out = nn.Sigmoid()(model(d))
        # out_np = out.cpu().detach().numpy()
        # y_pred = np.r_[y_pred,out_np]
        # 这是预测的结果最终代表的哪个类别
        y_pred_torch = torch.concat([torch.argmax(out, dim=1).cpu(), y_pred_torch])
        y_true_torch = torch.concat([l.cpu(), y_true_torch])

    y_pred = y_pred_torch.cpu().detach().numpy()
    y_true = y_true_torch.cpu().detach().numpy()

    # y_pred = np.argmax(y_pred,axis=1)

    # 如果需要one hot形式，那么就预测结果转为one hot，就是直接统计出来预测的是类1还是类0
    if one_hot:
        y_pred = np.eye(model._num_classes)[y_pred]

    # 把真实标签也返回
    if ground_truth == True:
        return y_pred, y_true

    return y_pred

def get_logits_torch(
    test_loader, model, device="cpu", middle_measure="mean", variance_adjustment=1, max_conf = 1 - 1e-16,
        min_conf = 0 + 1e-16 , label = None
):
    """Takes in a test dataloader + a trained model and returns the scaled logit values

    ...
    Parameters
    ----------
        test_loader : PyTorch Dataloader
            The Pytorch Dataloader for Dtest
        # 训练好的模型
        model : torch.nn.Module (PyTorch Neural Network Model)
            A trained model to be queried on the test data
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:0
        middle_measure : str
            # 参数 middle_measure 用于确定在去除数据中的离群值时使用的“中心”度量。这里的“中心”是指数据分布的中心位置，
            # 根据这个中心位置来确定数据的边界，进而识别并排除那些远离中心的离群值。
            # "mean": 使用数据的平均值作为中心度量。这意味着在计算标准差后，所有远离平均值的数据点
            # （超过 variance_adjustment指定的标准差数量）将被视为离群值并被去除。
            # "median": 使用数据的中位数作为中心度量。与平均值相比，中位数对极端值（离群值）不敏感，因此在数据分布不对称
            # 或含有离群值时，使用中位数作为中心度量可能更稳健。
            When removing outliters from the data, this is the
            "center" of the distribution that will be used.
            Options are ["mean", "median"]
        variance_adjustment : float
            The number of standard deviations away from the "center"
            we want to keep.
    ...
    Returns
    -------
        logits_arr : np.array
            An array containing the scaled model confidence values on the query set
    """
    # 要查询的样本总数
    n_samples = len(test_loader.dataset)
    # 初始化一个array来放结果
    logit_arr = np.zeros((n_samples, 1))
    # activation_dict = {}
        
    model = model.to(device)
    # 转为tensor
    y_prob = torch.Tensor([])
    y_test = torch.Tensor([])
    for d, l in test_loader:
        d = d.to(device)
        # 设为评估模式
        model.eval()
        with torch.no_grad():
            # 获得输出
            # 注意这是一批数据的输出
            out = model(d)
            # Get class probabilities
            # 在第二个维度上应用softmax
            # 我现在才发现他的模型训练好奇怪。训练的模型也就是model最后一层没有激活函数的。
            # 但是训练时候的标签还是1，好奇怪
            out = nn.functional.softmax(out, dim=1).cpu()
            # 沿着行拼接到一起
            y_prob = torch.concat([y_prob, out])
            y_test = torch.concat([y_test, l])
    # 转为numpy类型
    y_prob, y_test = np.array(y_prob), np.array(y_test, dtype=np.uint8)

    # print(y_prob.shape)
    # 这个是对y_prob中的每一行中每个元素应用比较,最终所有元素都为true或者false,
    # 一行代表一个样本,一个样本输出是两个值,是两个类别
    # sum表示对所有结果累加,如果大于1代表存在大于max_conf的,那么就要数值稳定了
    if np.sum(y_prob > max_conf):
        # 找到非零元素所在的坐标,也就是true的坐标,是二维的
        indices = np.argwhere(y_prob > max_conf)
        #             print(indices)
        # 对预测概率适当缩减,用于数值稳定性，防止计算 log 时取对数的值为负无穷
        for idx in indices:
            # r表示行数,c表示列号
            r, c = idx[0], idx[1]
            y_prob[r][c] = y_prob[r][c] - 1e-50

    # 如果概率太小,适当增加,用于数值稳定性
    # 其他的同理
    if np.sum(y_prob < min_conf):
        indices = np.argwhere(y_prob < min_conf)
        for idx in indices:
            r, c = idx[0], idx[1]
            y_prob[r][c] = y_prob[r][c] + 1e-50

    # 这一步是计算样本标签总数,但是实际上作者目前提供的实验都是二分类任务,不过这步检查还是可以的
    possible_labels = len(y_prob[0])
    # 把每一个样本的预测和真实标签组合起来
    for sample_idx, sample in enumerate(zip(y_prob, y_test)):

        # 预测的,真实的
        conf, og_label = sample
        # label表示需要我们关注的标签,也就是poison_class
        if(label == None):
            # 这步一般不会执行,
            label = og_label
        # 生成一个长度为类别数目的全true list
        selector = [True for _ in range(possible_labels)]
        # 把需要关注的置为false
        selector[label] = False

        # 这个就是用于计算置信度的 logit = log(p/(1-p))
        # 不过有问题，这个计算公式应该是针对sigmoid的输出的，但是作者确针对一个softmax的输出单独计算
        # 来当作logit，这一定是不匹配的，只能是一种近似计算。！！！！！！！！
        first_term = np.log(conf[label])
        # 至于为什么1-p不通过1-conf[label]计算在另一个论文里看到,计算会更稳定,效果更好
        second_term = np.log(np.sum(conf[selector]))

        logit_arr[sample_idx, 0] = first_term - second_term

    # print(logit_arr.shape)
    # 之前logit_arr是一个二维数组.虽然第二维的长度为1,现在重新铺平为1维的,为什么不一开始就弄成一维呢
    logit_arr = logit_arr.reshape(-1)

    # 用mean作为数据中心
    if middle_measure == "mean":
        middle = logit_arr.mean()
    # 用median作为数据中心
    elif middle_measure == "median":
        middle = np.median(logit_arr)

    # distinguish_type这个我都没见过就被注释了,删太多啦
    # if(distinguish_type == 'global_threshold'):
    # variance_adjustment用于控制要去掉的离群点的范围,越大,离群点越少
    # 保留logit值大于计算的阈值下限的logit
    logit_arr_filtered = logit_arr[
        logit_arr > middle - variance_adjustment * logit_arr.std()
    ]  # Remove observations below the min_range
    # 保留logit值低于计算的阈值上限的logit
    logit_arr_filtered = logit_arr_filtered[
        logit_arr_filtered < middle + variance_adjustment * logit_arr_filtered.std()
    ]  # Remove observations above max range

    return logit_arr_filtered

# 训练
def fit(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """
    # 转到指定设备上
    model = model.to(device)
    #初始化优化器
    # **optim_kwargs用来从参数字典中解包
    optimizer = optim_init(model.parameters(), **optim_kwargs)
    # 训练结果
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    # 打印提示信息
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:
        # 训练指定轮数
        # print(model.parameters())
        for epoch in range(1, epochs + 1):
            # 打印提示信息，可以统计时间
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0
            running_test_loss = 0
            running_test_acc = 0
            # 从dataloader中按批次加载数据训练
            for (inputs, labels) in dataloaders['train']:
                # 清零梯度
                optimizer.zero_grad()
                # 转换设备
                inputs = inputs.to(device)
                # print(inputs[:10,:])
                labels = labels.to(device)
                print(labels[:10])
                print(labels.shape)
                # 前向传播
                # 最后一层没有激活函数
                outputs = model.forward(inputs)
                print(outputs[:10, :])
                print(outputs.shape)
                # 计算损失
                # 注意，输出的第二维是两个数，因为两个类。但是label的第二维只有一维。
                # 具体原因就是，nn.CrossEntropyLoss会取out的第二维的最大值对应的
                # 索引作为模型预测的类别，然后计算这个预测类别与真实标签label之间的交叉熵损失
                # 且交叉熵损失函数起作用的只有1标签那个分类
                loss = criterion(outputs, labels)
                # 反向传播
                loss.backward()
                # 优化器优化
                optimizer.step()
                # 累加整个batch的损失
                running_train_loss += loss.item() * inputs.size(0)

            # 添加这个epoch的样本平均损失
            # 训练数据集对象
            train_error.append(running_train_loss / len(dataloaders['train'].dataset))

            # 如果不是第一次训练，且有早停策略
            if len(train_error) > 1 and early_stopping:
                # 如果最近两次loss下降的差值低于阈值，那么就要早停了
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    # 6位小数输出训练误差
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    # 如果不是只训练,那么额外返回测试误差和测试准确度,否则只返回训练误差
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error
            # 如果不是只训练,那就要计算测试损失和测试精度,问题来了.这里边根本没有任何测试集的影子.不知道是删减了还是啥
            if not train_only:
                test_loss.append(running_test_loss / len(dataloaders["test"].dataset))
                test_acc.append(running_test_acc / len(dataloaders["test"].dataset))
            # 打印时间和损失等信息
            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                if not train_only:
                    print(f"Test Error: {test_loss[-1]:.6}")
                    print(f"Test Accuracy: {test_acc[-1]*100:.4}%")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    # 如果训练被终止,打印当前信息
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
