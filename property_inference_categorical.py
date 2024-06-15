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

    for i in tqdm(range(num_shadow_models),
                  desc=f"Training {num_shadow_models} Shadow Models with {self._poison_percent * 100:.2f}% Poisoning"):

        if self._mini_verbose:
            print("-" * 10, f"\nModels {i + 1}")
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
            label=self._poison_class
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
            label=self._poison_class
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
    for i, poisoned_target_model in enumerate(
            tqdm(self._poisoned_target_models, desc=f"Querying Models and Running Distinguishing Test")):
        # 开始查询
        for query_trial in range(query_trials):
            # 啊,只是默认random查询.没有其他选择
            if query_selection.lower() == "random":
                Dtest_OH_sample_loader = training_utils.dataframe_to_dataloader(
                    # oversample_flag用于控制是否需要放回采样
                    self._Dtest_OH.sample(
                        n=self._num_queries, replace=oversample_flag, random_state=i + 1
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
                label=self._poison_class
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
                            f"Target is in t0 world with {M0_score / len(out_target) * 100:.4}% confidence"
                        )
                    # 那么就拿_target_worlds[i]和0去对比看看是否匹配，_target_worlds[i]代表真实的目标模型在哪个world上训练
                    correct_trials = correct_trials + int(
                        self._target_worlds[i] == 0
                    )

                # 否则就是反过来，一样的
                elif M0_score < M1_score:
                    if self._mini_verbose:
                        print(
                            f"Target is in t1 world {M1_score / len(out_target) * 100:.4}% confidence"
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