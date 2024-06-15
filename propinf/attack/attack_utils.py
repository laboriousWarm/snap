import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import propinf.data.ModifiedDatasets as data
from propinf.training import training_utils, models


class AttackUtil:
    def __init__(self, target_model_layers, df_train, df_test, cat_columns=None, verbose=True):
        if verbose:
            # 三引号用于定义多行字符串，或者包含引号、换行符等特殊字符的字符串。
            # 当字符串中包含换行时，使用三引号可以保持字符串的格式和布局。
            message = """Before attempting to run the property inference attack, set hyperparameters using
            1. set_attack_hyperparameters()
            2. set_model_hyperparameters()"""
            print(message)
        self.target_model_layers = target_model_layers
        self.df_train = df_train
        self.df_test = df_test
        self.cat_columns = cat_columns

        # Attack Hyperparams
        self._categories = None
        self._target_attributes = None
        self._sub_categories = None
        self._sub_attributes = None
        self._poison_class = None
        self._poison_percent = None
        self._k = None
        self._t0 = None
        self._t1 = None
        self._middle = None
        self._variance_adjustment = None
        self._num_queries = None
        self._nsub_samples = None
        self._ntarget_samples = None
        self._subproperty_sampling = False
        self._allow_subsampling = False
        self._allow_target_subsampling = False
        self._restrict_sampling = False
        self._pois_rs = None
        self._model_metrics = None

        # Model + Data Hyperparams
        self._layer_sizes = None
        self._num_classes = None
        self._epochs = None
        self._optim_init = None
        self._optim_kwargs = None
        self._criterion = None
        self._device = None
        self._tol = None
        self._verbose = None
        self._early_stopping = None
        self._dropout = None
        self._shuffle = None
        self._using_ce_loss = None
        self._batch_size = None
        self._num_workers = None
        self._persistent_workers = None

    def set_attack_hyperparameters(
        self,
        # 想攻击的属性名：
        categories=["race"],
        # 想攻击的属性值
        target_attributes=[" Black"],
        # 优化版本攻击的子集属性名
        sub_categories=["occupation"],
        # 优化版本攻击的子集属性值
        sub_attributes=[" Sales"],
        # 是否用优化版本的子集攻击
        subproperty_sampling=False,
        # 不明确现在
        restrict_sampling=False,
        # 想中毒的类别
        poison_class=1,
        # 中毒率
        poison_percent=0.03,
        # 指定了毒化数据集中特定类别的样本数量。如果未设置，将根据 poison_percent 计算
        k=None,
        # 两个参数用于生成特定比例的数据集，也就是两个不同world的数据集供猜测
        t0=0.1,
        t1=0.25,
        # 暂时不明确
        middle="median",
        variance_adjustment=1,
        # 训练阴影模型的数据集大小？？？？
        nsub_samples=1000,
        allow_subsampling=False,
        # 目标模型的样本大小
        ntarget_samples=1000,
        # 要训练几个目标模型
        num_target_models=25,
        allow_target_subsampling=False,
        # 随机数种子
        pois_random_seed=21,
        # 查询数量
        num_queries=1000,
    ):

        self._categories = categories
        self._target_attributes = target_attributes
        self._sub_categories = sub_categories
        self._sub_attributes = sub_attributes
        self._subproperty_sampling = subproperty_sampling
        self._restrict_sampling = restrict_sampling
        self._poison_class = poison_class
        self._poison_percent = poison_percent
        self._k = k
        self._t0 = t0
        self._t1 = t1
        self._middle = middle
        self._variance_adjustment = variance_adjustment
        self._num_queries = num_queries
        self._nsub_samples = nsub_samples
        self._allow_subsampling = allow_subsampling
        self._num_target_models = num_target_models
        self._ntarget_samples = ntarget_samples
        self._allow_target_subsampling = allow_target_subsampling
        self._pois_rs = pois_random_seed

    def set_shadow_model_hyperparameters(
        self,
        # 网络框架，默认是线性或者逻辑回归？因为就一层
        layer_sizes=[64],
        # 类别种类数，默认都是2分类
        num_classes=2,
        # 训练轮数
        epochs=10,
        # 默认adam优化器
        optim_init=optim.Adam,
        # 参数设置
        optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
        # 交叉熵损失函数
        criterion=nn.CrossEntropyLoss(),
        # 默认cpu训练
        device="cpu",
        # 损失函数低于阈值？
        tol=10e-7,
        # 母鸡
        verbose=True,
        # 目前看起来就是决定是否要打印一些提示信息
        mini_verbose=True,
        # 是否早停
        early_stopping=True,
        # 是否用dropout
        dropout=False,
        # 数据集默认打乱
        shuffle=True,
        # 是否用交叉熵损失函数，默认true
        using_ce_loss=True,
        # batch_size
        batch_size=1024,
        # 工作核心数量
        num_workers=8,
        # 母鸡
        persistent_workers=True,
    ):

        self._layer_sizes = layer_sizes
        self._num_classes = num_classes
        self._epochs = epochs
        self._optim_init = optim_init
        self._optim_kwargs = optim_kwargs
        self._criterion = criterion
        self._device = device
        self._tol = tol
        self._verbose = verbose
        self._mini_verbose = mini_verbose
        self._early_stopping = early_stopping
        self._dropout = dropout
        self._shuffle = shuffle
        self._using_ce_loss = using_ce_loss
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._activation_val = {}

    # 生成数据集
    def generate_datasets(self):
        """Generate all datasets required for the property inference attack"""

        # 目标世界，前边都是world0 ，后边是world1
        # 可以改成
        # self._target_worlds = np.concatenate(
        #   (np.zeros(self._num_target_models), np.ones(self._num_target_models))
        # )
        # 更高效。当然现在这点也没区别
        self._target_worlds = np.append(
            np.zeros(self._num_target_models), np.ones(self._num_target_models)
        )

        # print(self._target_worlds)

        # Generate Datasets
        if self._mini_verbose:
            print("Generating all datasets...")
        # 前四个数据集从训练集中划分的，目标和攻击者对半采样，后两个是测试集中采样的
        (
            self._D0_mo,
            self._D1_mo,
            self._D0,
            self._D1,
            self._Dp,
            self._Dtest,
        ) = data.generate_all_datasets(
            # 原始划分好的训练集
            self.df_train,
            # 原始划分好的测试集
            self.df_test,
            # 两个world的具有目标属性的数据的比例
            t0=self._t0,
            t1=self._t1,
            # 想攻击的属性名
            categories=self._categories,
            # 想攻击的属性值
            target_attributes=self._target_attributes,
            # 优化攻击版本中的子集攻击
            sub_categories=self._sub_categories,
            sub_attributes=self._sub_attributes,
            # 想要中毒的类别
            poison_class=self._poison_class,
            # 中毒率
            poison_percent=self._poison_percent,
            # 是否采用优化攻击
            subproperty_sampling=self._subproperty_sampling,
            # 是否限制采样，也就是把攻击者的训练数据集的部分提取出来
            restrict_sampling=self._restrict_sampling,
            # 是否输出提示信息
            verbose=self._verbose,
        )

        # 如果world0是没有符合目标属性的数据集，那么就直接把划分出来的中毒数据和测试集结合起来
        # 但是test是没有中毒的数据，那么标签不就混了吗，可能还是只需要数据，不需要标签吧。因为只是用来查询
        # 但是如果混起来，中毒数据也用于查询，难道不会影响分布吗，论文理论上是没有混合两者的
        if self._t0 == 0:
            self._Dtest = pd.concat([self._Dp, self._Dtest])

        #Changes-Harsh
        # Converting to poisoned class
        # self._Dtest["class"] = self._poison_class

        # 这个部分是跟中毒率相关。如果提供了中毒量大小，那么就计算一下中毒率，否则计算中毒量，两者肯定要提供一个的
        if self._k is None:
            self._k = int(self._poison_percent * len(self._D0_mo))
        else:
            self._poison_percent = self._k / len(self._D0_mo)

        # 如果中毒集合为空，感觉只有中毒率为0啊，或者其他的但是我现在也不知道。
        # 不过下边核心就是对数据集做一个one hot处理，其实一开始生成数据集那会也有相应的设置。但是默认为false。也就是不用one hot
        # 不过有个问题，之前在处理原始数据的时候，把最后一列的列名换为了label，那么这里仍然用class不会报错吗
        # 又回头看了一下，那个只有处理原始数据的时候直接生成one hot时才会替换class列为label列。那么看这里代码应该就是默认
        # 之前的one hot那步分支不会实现。
        # 更标准做法。应该是lass_label=self.df_train.columns[-1],
        if len(self._Dp) == 0:
            (
                _,
                self._D0_mo_OH,
                self._D1_mo_OH,
                self._D0_OH,
                self._D1_OH,
                self._Dtest_OH,
                self._test_set,
            ) = data.all_dfs_to_one_hot(
                [
                    self.df_train,
                    self._D0_mo,
                    self._D1_mo,
                    self._D0,
                    self._D1,
                    self._Dtest,
                    self.df_test,
                ],
                cat_columns=self.cat_columns,
                class_label="class",
            )
        # 两者区别无非就是有没有中毒数据集
        else:
            (
                _,
                self._D0_mo_OH,
                self._D1_mo_OH,
                self._D0_OH,
                self._D1_OH,
                self._Dp_OH,
                self._Dtest_OH,
                self._test_set,
            ) = data.all_dfs_to_one_hot(
                [
                    self.df_train,
                    self._D0_mo,
                    self._D1_mo,
                    self._D0,
                    self._D1,
                    self._Dp,
                    self._Dtest,
                    self.df_test,
                ],
                cat_columns=self.cat_columns,
                class_label="class",
            )
    # need_metrics: 布尔值，指示是否需要模型训练的详细指标
    # df_cv：交叉验证使用的 DataFrame，如果提供，将用于模型的交叉验证
    # 这个就是在中毒数据集和干净数据上训练模型
    def train_and_poison_target(self, need_metrics=False, df_cv=None):
        """Train target model with poisoned set if poisoning > 0"""

        # dataloader字典集合
        owner_loaders = {}
        # 2n个模型，none就是用来填充占位的，有时候可以用append就行。n表示目标模型的数量，2表示两个world
        self._poisoned_target_models = [None] * self._num_target_models * 2
        # 数据的输入维度，-1是因为去掉标签列
        input_dim = len(self._D0_mo_OH.columns) - 1

        # 不允许对目标模型的训练集采样，也就是说每次训练目标模型都是用的相同的数据
        if self._allow_target_subsampling == False:
            # 训练数据集的大小
            self._ntarget_samples = self._D0_mo_OH.shape[0]
            # 如果中毒数据集大小为空，那就从初始的D0里边选择数据集来做训练集
            if len(self._Dp) == 0:
                poisoned_D0_MO = self._D0_mo_OH.sample(
                    n=self._ntarget_samples, random_state=21
                )
            # 否则就采样（1-p）的干净数据，这里解释了我的疑惑，真正数据集的划分是在这里做的。但是他的比例确实还是有点不准确
            else:
                # Changes
                clean_D0_MO = self._D0_mo_OH.sample(
                    n=int((1 - self._poison_percent) * self._ntarget_samples),
                    random_state=21,
                )
                # 如果有足够多的原始中毒样本，那么直接和干净样本拼接变成训练集
                if (
                    int(self._poison_percent * self._ntarget_samples) <= self._Dp_OH.shape[0]
                ):

                    poisoned_D0_MO = pd.concat(
                        [
                            clean_D0_MO,
                            # 采样np个有毒数据
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    # 否则有放回采样有毒数据
                    poisoned_D0_MO = pd.concat(
                        [
                            clean_D0_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )
        # 在world0上训练n个目标模型
        """Trains half the target models on t0 fraction"""
        # for i in tqdm(range(self._num_target_models), desc = "Training target models with frac t0"):
        for i in tqdm(range(self._num_target_models), desc=f"Training Target Models with {self._poison_percent*100:.2f}% poisoning"):
            # 如果允许对目标模型的训练集采样
            if self._allow_target_subsampling == True:
                # 如果中毒数据集为空，直接_D0_mo_OH用做训练集
                # 随机数种子在变，每次的训练集有所不同
                if len(self._Dp) == 0:
                    poisoned_D0_MO = self._D0_mo_OH.sample(
                        n=self._ntarget_samples, random_state=i + 1
                    )
                # 否则和中毒拼接起来
                else:

                    poisoned_D0_MO = pd.concat(
                        [
                            # 随机数种子在变，每次的训练集有所不同
                            self._D0_mo_OH.sample(
                                n=int(
                                    (1 - self._poison_percent) * self._ntarget_samples
                                ),
                                random_state=i + 1,
                            ),
                            # 奇怪，这里又不考虑，原始中毒数据不够了，且这里的随机数种子是固定的
                            # 也就是在训练目标模型的时候，使用的都是同样的中毒数据集
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                            ),
                        ]
                    )
            # 生成训练集的dataloader
            owner_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0_MO,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )
            # 如果提供了目标模型的参数，默认是32，16，输出类别都是2，作者搞的都是2分类任务
            # 这部就是设置目标模型
            # 注意因为默认用交叉熵损失函数，所以模型最后一层之后没有添加softmax激活函数
            # 因为交叉熵损失函数自己有一步计算softmax。所以送给他的输出必须是原始的logit
            if len(self.target_model_layers) != 0:
                # print(self.target_model_layers)
                target_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )
            # 否则就用最简单的逻辑回归模型作为目标模型
            else:
                target_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )
            # 打印提示信息
            if self._mini_verbose:
                print("-" * 10, "\nTarget Model")

            # 训练，并把模型记录到_poisoned_target_models中
            self._poisoned_target_models[i], _ = training_utils.fit(
                dataloaders=owner_loaders,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

        # 上边是对world0的处理,下边是对world1的处理,一模一样,我就不讲了,但是确实不理解的是什么时候中毒率才是0呢
        # 且目前也没看到查询
        if self._allow_target_subsampling == False:
            self._ntarget_samples = self._D0_mo_OH.shape[0]
            if len(self._Dp) == 0:
                # poisoned_D1_MO = self._D1_mo_OH.copy()
                poisoned_D1_MO = self._D1_mo_OH.sample(
                    n=self._ntarget_samples, random_state=21
                )

            else:
                clean_D1_MO = self._D1_mo_OH.sample(
                    n=int((1 - self._poison_percent) * self._ntarget_samples),
                    random_state=21,
                )
                # Changes
                if (
                    int(self._poison_percent * self._ntarget_samples)
                    <= self._Dp_OH.shape[0]
                ):
                    poisoned_D1_MO = pd.concat(
                        [
                            clean_D1_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    poisoned_D1_MO = pd.concat(
                        [
                            clean_D1_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

        """Trains half the target models on t1 fraction"""
        for i in range(self._num_target_models, 2 * self._num_target_models):
            # for i in tqdm(range(self._num_target_models, 2*self._num_target_models), desc = "Training target models with frac t1"):

            if self._allow_target_subsampling == True:
                if len(self._Dp) == 0:
                    poisoned_D1_MO = self._D1_mo_OH.sample(
                        n=self._ntarget_samples, random_state=i + 1
                    )

                else:
                    poisoned_D1_MO = pd.concat(
                        [
                            self._D1_mo_OH.sample(
                                n=int(
                                    (1 - self._poison_percent) * self._ntarget_samples
                                ),
                                random_state=i + 1,
                            ),
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                            ),
                        ]
                    )

            owner_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D1_MO,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            if len(self.target_model_layers) != 0:
                target_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

            else:
                target_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._mini_verbose:
                print("-" * 10, "\nTarget Model")

            self._poisoned_target_models[i], _ = training_utils.fit(
                dataloaders=owner_loaders,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

    # 执行属性推断攻击
    def property_inference_categorical(
        self,
        # 每个world需要的阴影模型数量
        num_shadow_models=1,
        # 查询轮数
        query_trials=1,
        # 随机挑选查询数据集
        query_selection="random",
        # 区分两个world的方法
        distinguishing_test="median",
    ):
        """Property inference attack for categorical data. (e.g. Census, Adults)

        ...
        Parameters
        ----------
            num_shadow_models : int
                The number of shadow models per "world" to use in the attack
            query_trials : int
                Number of times we want to query num_queries points on the target
            distinguishing_test : str
                The distinguishing test to use on logit distributions
                Options are the following:

                # 戛然而止,真的删减了.哎
                # 中位数的中位数：首先，这个方法会计算两个影子模型（分别代表两个不同的“世界”或数据分布）在测试数据集上的预测输出的中位数。
                # 然后，计算这两组中位数的平均值，得到一个阈值（threshold）。
                # 判断：接下来，使用这个阈值来判断目标模型的预测信心大多数位于哪一侧。如果目标模型的预测信心多数高于这个阈值，
                # 可以推断它更可能来自与较高中位数相对应的“世界”；如果多数低于阈值，则更可能来自另一个“世界”。
                "median" : uses the middle of medians as a threshold and checks on which side of the threshold
                           the majority of target model prediction confidences are on
                # KL散度（Kullback-Leibler divergence）：这个方法使用KL散度来衡量两个概率分布之间的差异。
                # 在这种情况下，两个分布是目标模型和影子模型在相同测试数据集上的预测结果。
                # 相似性测量：KL散度可以衡量一个分布转换到另一个分布所需的信息量。在
                # 属性推断的上下文中，它被用来衡量目标模型的预测分布与两个影子模型预测分布的相似度。
                # 决策依据：通过比较目标模型与两个影子模型的KL散度，可以推断目标模型更接近哪个影子模型的“世界”。
                # 较小的KL散度意味着较低的分布差异，表明目标模型可能在与该影子模型相同的数据分布上训练。
                "divergence" : uses KL divergence to measure the similarity between the target model
                               prediction scores and the

        ...
        Returns
        ----------
            out_M0 : np.array
                Array of scaled logit values for M0
            out_M1 : np.array
                Array of scaled logit values for M1
            logits_each_trial : list of np.arrays
                Arrays of scaled logit values for target model.
                Each index is the output of a single query to the
                target model
            predictions : list
                Distinguishing test predictions; 0 if prediction
                is t0, 1 if prediction is t1
            correct_trials : list
                List of booleans dentoting whether query trial i had
                a correct prediction
        """

        # Train multiple shadow models to reduce variance
        if self._mini_verbose:
            print("-" * 10, "\nTraining Shadow Models...")
        # 两个world的dataloader
        D0_loaders = {}
        D1_loaders = {}
        # 如果不允许子采样,那就直接采样一次不变.且直接用_D0_OH数据集,这个环节讲解过
        if self._allow_subsampling == False:

            self._nsub_samples = self._D0_OH.shape[0]

            # print("Size of Shadow model dataset: ", self._nsub_samples)

            if len(self._Dp) == 0:
                poisoned_D0 = self._D0_OH.sample(n=self._nsub_samples, random_state=21)
                poisoned_D1 = self._D1_OH.sample(n=self._nsub_samples, random_state=21)

            else:
                clean_D0 = self._D0_OH.sample(
                    n=int((1 - self._poison_percent) * self._nsub_samples),
                    random_state=21,
                )
                clean_D1 = self._D1_OH.sample(
                    n=int((1 - self._poison_percent) * self._nsub_samples),
                    random_state=21,
                )

                if (
                    int(self._poison_percent * self._nsub_samples)
                    <= self._Dp_OH.shape[0]
                ):
                    # Changes
                    poisoned_D0 = pd.concat(
                        [
                            clean_D0,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                    poisoned_D1 = pd.concat(
                        [
                            clean_D1,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    poisoned_D0 = pd.concat(
                        [
                            clean_D0,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )
                    poisoned_D1 = pd.concat(
                        [
                            clean_D1,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )
            # 不允许子采样的时候,训练数据集就只有一个,不会变
            D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            D1_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D1,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )
        # 生成查询集合
        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            self._Dtest_OH,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )
        # 输入维度
        input_dim = len(self._D0_mo_OH.columns) - 1

        # 初始化为空
        out_M0 = np.array([])
        out_M1 = np.array([])

        for i in tqdm(range(num_shadow_models), desc=f"Training {num_shadow_models} Shadow Models with {self._poison_percent*100:.2f}% Poisoning"):

            if self._mini_verbose:
                print("-" * 10, f"\nModels {i+1}")
            # 跟训练目标模型的区别就是之前两个world是分开设定的,现在是一起弄的,减少了一些重复操作,但是核心跟上个函数一样,就不说了
            if len(self._layer_sizes) != 0:
                M0_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self._layer_sizes,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

                M1_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self._layer_sizes,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )
            else:
                M0_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

                M1_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._allow_subsampling == True:

                if len(self._Dp) == 0:
                    poisoned_D0 = self._D0_OH.sample(n=self._nsub_samples)
                    poisoned_D1 = self._D1_OH.sample(n=self._nsub_samples)

                else:

                    if self._allow_target_subsampling == True:

                        poisoned_D0 = pd.concat(
                            [
                                self._D0_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples),
                                    random_state=self._pois_rs,
                                ),
                            ]
                        )

                        poisoned_D1 = pd.concat(
                            [
                                self._D1_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples),
                                    random_state=self._pois_rs,
                                ),
                            ]
                        )
                    else:
                        poisoned_D0 = pd.concat(
                            [
                                self._D0_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples)
                                ),
                            ]
                        )

                        poisoned_D1 = pd.concat(
                            [
                                self._D1_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples)
                                ),
                            ]
                        )

                D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D0,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                )

                D1_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D1,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                )

            # 训练world0的阴影模型
            M0_trained, _ = training_utils.fit(
                dataloaders=D0_loaders,
                model=M0_model,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            # 训练world1的
            M1_trained, _ = training_utils.fit(
                dataloaders=D1_loaders,
                model=M1_model,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            # 根据模型输出计算logit,并用_variance_adjustment控制去掉离群点logit,这是论文没提的技巧啊
            # 这样图就更好看
            out_M0_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M0_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                # label是想要用于计算置信度的那个标签
                label = self._poison_class
            )
            # 输出添加进来
            out_M0 = np.append(out_M0, out_M0_temp)

            # 跟上边一样
            out_M1_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M1_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            out_M1 = np.append(out_M1, out_M1_temp)

            # 打印提示信息,统计所有logit的均值和方差,标准差,中位数
            if self._verbose:
                print(
                    f"M0 Mean: {out_M0.mean():.5}, Variance: {out_M0.var():.5}, StDev: {out_M0.std():.5}, Median: {np.median(out_M0):.5}"
                )
                print(
                    f"M1 Mean: {out_M1.mean():.5}, Variance: {out_M1.var():.5}, StDev: {out_M1.std():.5}, Median: {np.median(out_M1):.5}"
                )

        # 如果是用median方法作为区分测试,那么阈值就是两个不同分布(t0, t1)的中位数的中间值
        # 也就是代表两个高斯分布的交叉点的近似,这个时候,分类误差最小
        if distinguishing_test == "median":
            midpoint_of_medians = (np.median(out_M0) + np.median(out_M1)) / 2
            thresh = midpoint_of_medians
        # 作者去掉了利用kl散度近似的情况讨论,在一开始说明的时候介绍了,论文里记得也没提,也就是说可能是作者的实验探索

        # 打印阈值信息
        if self._verbose:
            print(f"Threshold: {thresh:.5}")

        # Query the target model and determine
        # 用于统计一共成功推断了几次
        correct_trials = 0

        # 打印总查询次数,每次查询样本数
        if self._mini_verbose:
            print(
                "-" * 10,
                f"\nQuerying target model {query_trials} times with {self._num_queries} query samples",
            )

        # 根据实际情况决定是否需要过采样
        oversample_flag = False
        # 如果查询点数目大于查询样本的大小,那么就需要过采样,干净主要是控制打印信息,然后也减少了后边一直判断
        if self._num_queries > self._Dtest_OH.shape[0]:
            oversample_flag = True
            print("Oversampling test queries")

        # 对每个目标模型查询
        for i, poisoned_target_model in enumerate(tqdm(self._poisoned_target_models, desc=f"Querying Models and Running Distinguishing Test")):
            # 开始查询
            for query_trial in range(query_trials):
                # 啊,只是默认random查询.没有其他选择
                if query_selection.lower() == "random":
                    Dtest_OH_sample_loader = training_utils.dataframe_to_dataloader(
                        # oversample_flag用于控制是否需要放回采样
                        self._Dtest_OH.sample(
                            n=self._num_queries, replace=oversample_flag, random_state = i+1
                        ),
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        using_ce_loss=self._using_ce_loss,
                    )
                else:
                    print("Incorrect Query selection type")

                # 对查询点在目标模型上查询,然后计算logit
                out_target = training_utils.get_logits_torch(
                    Dtest_OH_sample_loader,
                    poisoned_target_model,
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )

                # 打印统计信息
                if self._verbose:
                    print("-" * 10)
                    print(
                        f"Target Mean: {out_target.mean():.5}, Variance: {out_target.var():.5}, StDev: {out_target.std():.5}, Median: {np.median(out_target):.5}\n"
                    )

                # 开始区分是world0还是world1
                """ Perform distinguishing test"""
                if distinguishing_test == "median":
                    # 统计目标模型上查询出来的logit大于阈值和低于阈值的数目
                    # world0的logit分布在右侧,那么选的就是大于阈值的
                    M0_score = len(out_target[out_target > thresh])
                    # world1在左侧，那么选的就是低于阈值的
                    M1_score = len(out_target[out_target < thresh])
                    if self._verbose:
                        print(f"M0 Score: {M0_score}\nM1 Score: {M1_score}")

                    # 如果大部分符合world0，说明判断当前world为0
                    if M0_score >= M1_score:
                        if self._mini_verbose:
                            print(
                                f"Target is in t0 world with {M0_score/len(out_target)*100:.4}% confidence"
                            )
                        # 那么就拿_target_worlds[i]和0去对比看看是否匹配，_target_worlds[i]代表真实的目标模型在哪个world上训练
                        correct_trials = correct_trials + int(
                            self._target_worlds[i] == 0
                        )

                    # 否则就是反过来，一样的
                    elif M0_score < M1_score:
                        if self._mini_verbose:
                            print(
                                f"Target is in t1 world {M1_score/len(out_target)*100:.4}% confidence"
                            )

                        correct_trials = correct_trials + int(
                            self._target_worlds[i] == 1
                        )

        # 目前只提供了一个median的分支，没有提供kl散度的判断，下边就是返回实验结果
        if distinguishing_test == "median":
            return (
                out_M0,
                out_M1,
                thresh,
                correct_trials / (len(self._target_worlds) * query_trials),
            )