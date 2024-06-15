import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_adult_columns():
    # 获取adult数据集中的连续项和离散项

    # 原始数据有14个属性，但是作者去掉了fnlwgt和education两个属性
    column_names = [
        "age",
        "workclass",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]

    cont_columns = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_columns = sorted(list(set(column_names).difference(cont_columns)))
    # cat_columns = sorted(set(column_names).difference(cont_columns))
    # 不用转为list也行.sorted的函数接收可迭代对象作为参数.不过取两个列表的常见操作就是转换为集合取差
    # 突然发现，这里只有离散列是做了sort的
    # cat_columns = list(set(column_names).difference(cont_columns))

    return cat_columns, cont_columns


def get_census_columns():
    """Returns names of categorical and continuous columns for census dataset"""

    column_names = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-stat",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-stat",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-stat",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-stat",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
    ]

    cont_columns = [
        "age",
        "wage-per-hour",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "instance-weight",
        "num-persons-worked-for-employer",
        "weeks-worked-in-year",
    ]
    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    return cat_columns, cont_columns


# 这个函数就是用来控制某种类别的数据在总的数据集中的比例，但是还有缺陷,比如no数量大于other_data数量
# 这时候可以过采样试试
def generate_class_imbalance(data, target_class, target_ratio):
    # 统计目标类别的数量，在当前设置中为正类的数量
    Nt = sum(data["class"] == target_class)
    # 想满足target_class占总的比率target_ratio，需要设置非target_class的数量
    No = (1 - target_ratio) * Nt / target_ratio

    # 获取哪些行是正类，哪些行是负类,tgt_idx是一个布尔类型的series
    tgt_idx = data["class"] == target_class
    # tgt_data是所有class是target_class的数据
    tgt_data = data[tgt_idx]
    # 取反也就是获取所有非target_class的数据
    other_data = data[~tgt_idx]
    # train_test_split是一个分割数据集的函数，也就是从原始的other_data中划分出来No个数据
    # 但是没考虑No数量大于other_data的情况，也就是说作者设置的实验都默认不会发生这种情况
    other_data, _ = train_test_split(
        other_data, train_size=No / other_data.shape[0], random_state=21
    )

    # 拼接完后就是target_class数据符合target_ratio的数据
    data = pd.concat([tgt_data, other_data])
    # sample(frac=1)这个相当于对全部数据随机打乱一遍，注意，此时我们的data的索引也是打乱的，也就是说不是从0开始的
    # 那么需要reset_index()，默认的drop为false，会把原始的索引当作新的一列，然后再从0创建索引。
    # drop为true的时候，抛弃原来的索引
    data = data.sample(frac=1).reset_index(drop=True)

    return data

# 就是把一个数据集中你想要的一个属性占比调成你想要的
def v2_fix_imbalance(
    # 要调整的数据集
    df,
    # 目标属性在数据集中的比例，例如女性
    target_split=0.4,
    # 想调整的属性名
    categories=["sex"],
    # 想调整的属性值
    target_attributes=[" Female"],
    # 随机数种子
    random_seed=21,
    # 母鸡
    return_indices=False,
):
    """Corrects a data set to have a target_split percentage of the target attributes

    ...
    Parameters
    ----------
        df : Pandas Dataframe
            The dataset
        target_split : float
            The desired proportion of the subpopulation within the dataset
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_indices : bool
            If True, returns the indices to drop instead of the modified dataframe

    ...
    Returns
    -------

    df : Pandas Dataframe
        The dataset with a target_split/(1-target_split) proportion of the subpopulation
    """

    # 缺点target_split的合法性
    assert target_split >= 0 and target_split <= 1, "target_split must be in (0,1)"

    # 如果没有目标，那就直接返回原始数据就行，是无意义的调用本函数
    if len(categories) == 0 or len(target_attributes) == 0:
        return df

    # 创建一个副本，可能是希望当前修改不影响原始数据集，不copy的话就是引用，那么所有修改都会影响原始值
    # 原始副本用的地方很多，所以不想修可能
    df = df.copy()

    # 设置随机数种子
    np.random.seed(random_seed)

    # 这个列表中的每个元素都是一个布尔序列（Pandas Series），代表数据集中每一行是否符合特定的属性值
    indices_with_each_target_prop = []

    # 对所有属性名和值的组合的遍历
    for category, target in zip(categories, target_attributes):
        # 对每一对属性名和值的组合，向indices_with_each_target_prop中添加一个bool list
        # 这个bool list中每一个元素代表，对应df中的行是否是属性值为target的行
        # 这个用于后边筛选出符合要求的数据，进而调整比例
        indices_with_each_target_prop.append(df[category] == target)

    # 这个的最终结果就是一个列表，每个元素代表数据集的一行是否全部匹配给定的属性名和值的组合
    indices_with_all_targets = np.array(
        # zip(*indices_with_each_target_prop)]
        # 首先*indices_with_each_target_prop，把大的外层的list解开，然后通过zip再把这些小的list中的元素
        # 按照序号对应起来组成一个元组，最后再zip拼成一个大list，这个大list的一个元素就是一个元组，
        # 元组内元素就是数据的某一行是否满足名=特定值
        # all函数就是检查可迭代对象是否都为true，把元组聚合为一个true或者false
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )

    # 挑选出符合要求的所有行
    subpop_df = df[indices_with_all_targets]
    # 剩下不符合要求的数据
    remaining_df = df[~indices_with_all_targets]

    # 不符合要求的数据有多少个
    rem_samples = remaining_df.shape[0]
    # 根据不符合要求的数据数目来计算出，要满足target_split，需要采样多少个符合要求的数据
    # rem_samples / (1 - target_split)就是满足target_split比例的数据总数
    subpop_samples = int(target_split * rem_samples / (1 - target_split))

    # 如果符合要求的数据数目是大于要采样的，按摩就正常采样然后拼接出来就行
    # 获取的就是符合要求的数据
    if subpop_samples <= subpop_df.shape[0]:
        df = pd.concat(
            # replace=False（默认值）：这意味着在抽取样本时，不允许重复。无放回抽样
            [remaining_df, subpop_df.sample(n=subpop_samples, random_state=random_seed)]
        )
    else:
        # 否则，说明原始数据中符合要求的目标属性数据小于要采样的数目，那么就要过采样了
        # print("oversampling")
        df = pd.concat(
            [
                remaining_df,
                subpop_df.sample(
                    # replace=True：这意味着在抽取样本时，允许重复。有放回抽样
                    n=subpop_samples, replace=True, random_state=random_seed
                ),
            ]
        )

    # 又是全部打乱然后重置索引。因为之前是满足要求的在一块，不满足要求的数据在一块，拼接之后，还是直接两部分，所以需要打乱
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def generate_subpopulation(
    df, categories=[], target_attributes=[], return_not_subpop=False
):
    """Given a list of categories and target attributes, generate a dataframe with only those targets
    ...
    Parameters
    ----------
        df : Pandas Dataframe
            A pandas dataframe
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_not_subpop : bool
            If True, also return df/subpopulation

    ...
    Returns
    -------
        subpop : Pandas Dataframe
            The dataframe containing the target subpopulation
        not_subpop : Pandas Dataframe (optional)
            df/subpopulation
    """
    # 跟之前分析的函数一样。最终获得了一个list，每个元素代表当前数据的某一行是否满足所有的属性要求
    indices_with_each_target_prop = []

    for category, target in zip(categories, target_attributes):

        indices_with_each_target_prop.append(df[category] == target)

    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )

    # 如果为真，那么不止返回符合要求的数据，还把剩下的不符合的数据也返回
    if return_not_subpop:
        return df[indices_with_all_targets].copy(), df[~indices_with_all_targets].copy()
    else:
        # 默认这个分支，只返回符合要求的数据
        return df[indices_with_all_targets].copy()


def generate_all_datasets(
    train_df,
    test_df,
    t0=0.1,
    t1=0.5,
    categories=["race"],
    target_attributes=[" White"],
    sub_categories=["occupation"],
    sub_attributes=[" Sales"],
    # 攻击类别
    poison_class=1,
    # 控制毒化集的大小
    k=None,
    poison_percent=None,
    verbose=True,
    # 是否允许自定义类别频率
    allow_custom_freq=False,
    # 是的话，频率占比
    label_frequency=0.5,
    # 是否使用优化攻击
    subproperty_sampling=False,
    # 母鸡
    restrict_sampling=False,
    # 随机数种子
    random_state=21,
):
    """Generates the model owner's dataset (D_mo), the adversary's datasets (D0, D1), and
    the poisoned dataset (Dp)

        ...
        Parameters
        ----------
            train_df : Pandas Dataframe
                The train set
            test_df : Pandas Dataframe
                The validation set or some dataset that is disjoint from train_df,
                but drawn from the same distribution
            mo_frac: float
                Setting the proportion of of the subpopulation model owner's  to mo_frac
            t0 : float
                Lower estimate for proportion of the subpopulation in model owner's
                dataset
            t1 : float
                Upper estimate for proportion of the subpopulation in model owner's
                dataset
            categories : list
                Column names for each attribute
            target_attributes : list
                Labels to create subpopulation from.
                Ex. The subpopulation will be df[categories] == attributes
            poison_class : int
                The label we want our poisoned examples to have
            k : int
                The number of points in the poison set
            poison_percent : float
                [0,1] value that determines what percentage of the
                total dataset (D0) size we will make our poisoning set
                Note: Must use either k or poison_percent
            verbose : bool
                If True, reports dataset statistics via a print-out
            return_one_hot : bool
                If True returns dataframes with one-hot-encodings
            cat_columns : list
                The list of columns with categorical features (only used if
                return_one_hot = True)
            cont_columns : list
                The list of columns with continuous features (only used if
                return_one_hot = True)

        ...
        Returns
        -------
            D0_mo : Pandas Dataframe
                The model owner's dataset with t0 fraction of the target subpopulation
            D1_mo : Pandas Dataframe
                The model owner's dataset with t1 fraction of the target subpopulation
            D0 : Pandas Dataframe
                The adversary's dataset with t0 fraction of the target subpopulation
            D1 : Pandas Dataframe
                The adversary's dataset with t0 fraction of the target subpopulation
            Dp : Pandas Dataframe
                The adversary's poisoned set
            Dtest : Pandas Dataframe
                The adversary's query set
    """

    # 对t0，t1做判断，保证在[0,1)
    assert t0 >= 0 and t0 < 1, "t0 must be in [0,1)"
    assert t1 >= 0 and t1 < 1, "t1 must be in [0,1)"

    np.random.seed(random_state)
    # 拿出来原始训练集的所有索引
    all_indices = np.arange(0, len(train_df), dtype=np.uint64)
    # 打乱索引，这样就相当于实现随机采样
    np.random.shuffle(all_indices)

    # 把原始的训练数据集划分成两半，一半是目标模型的，一半是攻击者的,注意这是随机划分，那么索引也打乱了不是从0开始
    D_mo, D_adv = train_test_split(train_df, test_size=0.5, random_state=random_state)

    # 去掉原始索引，重新从0开始
    D_mo = D_mo.reset_index(drop=True)
    D_adv = D_adv.reset_index(drop=True)

    # 如果允许自定义类别频率，那么就调整数据集中的想要的某种标签比如1(收入大于50K)在数据集中的频率为label_frequency
    # 下边分别是调整目标模型的数据集和攻击者的数据集中指定属性的比例
    if allow_custom_freq == True:

        D_mo = v2_fix_imbalance(
            D_mo,
            target_split=label_frequency,
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )

        D_adv = v2_fix_imbalance(
            D_adv,
            target_split=label_frequency,
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )

    # 如果为真就打印提示信息
    if verbose:
        # 因为最后一列就是0，1，那么sum就是所有标签为1的数据行数，那么也就代表1标签的比例
        # 下边可以改成label_split_mo = D_mo.iloc[:,-1].sum() / len(D_mo)
        # 或者label_split_mo = (D_mo.iloc[:,-1]==1).sum() / len(D_mo)更加直观
        label_split_mo = sum(D_mo[D_mo.columns[-1]]) / len(D_mo)
        label_split_adv = sum(D_adv[D_adv.columns[-1]]) / len(D_adv)
        print(
            f"The model owner has {len(D_mo)} total points with {label_split_mo*100:.4}% class 1"
        )
        print(
            f"The adversary has {len(D_adv)} total points with {label_split_adv*100:.4}% class 1"
        )

    # 上边那个是调整特定标签数据的比例，现在调整的是子属性的比例，这个函数应该是默认满足要求的数据占少数，不满足的是多数。
    # 从上一步调整好的总体数据中分别采样出两个world的数据集
    D0_mo = v2_fix_imbalance(
        D_mo,
        target_split=t0,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    D1_mo = v2_fix_imbalance(
        D_mo,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )
    # 从上一步中采样出符合要求的数据中调整攻击者的两个world的训练集
    D0 = v2_fix_imbalance(
        D_adv,
        target_split=t0,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    D1 = v2_fix_imbalance(
        D_adv,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    )

    # 上一步调整过的数据集一定数目不等，因为获得的不满足要求的数据数目是相同的，那么计算出的需要的满足要求的数据数目一定不同
    # 因为t0，t1不同，但是需要两个world的训练集大小相等。
    # 作者就对大数据集无放回采样到小数据的规模。但是这个有问题，你这么采样，重复的肯定太多了，不如
    # D0_1 = D0.sample(n=len(D1-D0), random_state=random_state)
    # D0 = pd.pd.concat([D0, D0_1])
    # 但是这样也有一样的问题，过采样之后，数据的比例就变了不再是原来符合t0的world了。不知道作者是不知道啊，还是错了
    # 不如一开始就确定数据集的大小，根据数据集大小来确保采样特定比例的数据
    print(len(D0), len(D1), len(D0_mo), len(D1_mo))
    if len(D0) > len(D1):
        D0 = D0.sample(n=len(D1), random_state=random_state)
    elif len(D1) > len(D0):
        D1 = D1.sample(n=len(D0), random_state=random_state)

    if len(D0_mo) > len(D1_mo):
        D0_mo = D0_mo.sample(n=len(D1_mo), random_state=random_state)

    elif len(D1_mo) > len(D0_mo):
        D1_mo = D1_mo.sample(n=len(D0_mo), random_state=random_state)

    # 这步也是一样的问题，因为一开始目标模型和攻击者的数据集是随机划分的，所以计算出来的，不符合要求的数据是不同的，
    # 那么最终算出来的数据大小也是不同的。所以需要调整为一样大小，但是这么一调比例就不对，也不知道怎么成功推断的
    # 作者既然能意识到这个大小不同的问题，按说这个应该也可也啊，感觉像是改了代码。因为这中间是有些修改的，有一些明显、‘、
    # 被删除的。哎。不过自己弄的时候，最好就按照目标大小去采样就好了
    if len(D0_mo) > len(D0):
        D0_mo = D0_mo.sample(n=len(D0), random_state=random_state)
        D1_mo = D1_mo.sample(n=len(D0), random_state=random_state)

    elif len(D0) > len(D0_mo):
        D0 = D0.sample(n=len(D0_mo), random_state=random_state)
        D1 = D1.sample(n=len(D0_mo), random_state=random_state)

    # 打印提示信息
    if verbose:
        print(f"The model owner's dataset has been downsampled to {len(D0_mo)} points")

    if verbose:
        # 统计d0中正类的数目
        label_split_d0 = sum(D0[D0.columns[-1]]) / len(D0)
        # d1中正类的数目
        label_split_d1 = sum(D1[D1.columns[-1]]) / len(D1)
        # 生成D0中所有满足要求的数据
        D0_subpop = generate_subpopulation(
            D0, categories=categories, target_attributes=target_attributes
        )
        # D1中满足要求的数据
        D1_subpop = generate_subpopulation(
            D1, categories=categories, target_attributes=target_attributes
        )
        # 这边一打印，肯定能看出来不对劲呀。很奇怪。我跑了一遍发现目标数目虽然有差距，但是不是很大，我分析的也没错啊
        # 可能因为数据量不是很小，所以随机的结果还是不错的，没有太大的偏差
        print(
            f"D0 has {len(D0)} points with {len(D0_subpop)} members from the target subpopulation and {label_split_d0*100:.4}% class 1"
        )
        print(
            f"D1 has {len(D1)} points with {len(D1_subpop)} members from the target subpopulation and {label_split_d1*100:.4}% class 1"
        )

    # 计算需要的有毒数据的大小。这里有问题，论文中说的是中毒数据占全部数据（中毒+干净数据）的比例为p
    # 这里直接用干净数据作为n，来计算np，存在了近似，因为p比较小，但是这样还是适当放大了数据的中毒率啊。
    # 我目前的理解就是这样，源代码存在一些不那么严谨的地方。后边发现，在后续的代码中作者又重新采样了，解决了这个问题
    # 虽然这里是if判断，但是在调用这个函数的地方，他只用了p没提供k，所以一定会执行这个循环，可能在未公开代码有别的实现吧
    # 虽然他的攻击类根本没有提供k这个参数
    if poison_percent is not None:
        k = int(poison_percent * len(D0_mo))

    # 允许优化攻击，也就是利用子类
    if subproperty_sampling == True:
        # 生成中毒数据集，利用的是测试集的数据，也就是说中毒数据是和训练数据不相交的
        Dp, Dtest = generate_Dp(
            test_df,
            categories=categories + sub_categories,
            target_attributes=target_attributes + sub_attributes,
            k=k,
            poison_class=poison_class,
            random_state=random_state,
        )
    else:
        Dp, Dtest = generate_Dp(
            test_df,
            categories=categories,
            target_attributes=target_attributes,
            k=k,
            poison_class=poison_class,
            random_state=random_state,
        ) #普通版本攻击

    # 如果没有测试集，那么直接拿有毒数据集作为测试集，很奇怪啊。中毒的标签是假的啊，可能需要的只是数据，跟标签无关
    # 标签只影响训练，不影响查询，查询logit不需要标签
    if len(Dtest) == 0:
        Dtest = Dp.copy()

    # 统计信息
    # 但是这里没有考虑使用优化攻击的情况，只适用于普通版本
    if verbose:
        subpop = generate_subpopulation(
            test_df, categories=categories, target_attributes=target_attributes
        )
        print(
            f"The poisoned set has {k} points sampled uniformly from {len(subpop)} total points in the subpopulation"
        )

    # 如果限制采样，感觉只是额外统计了一些符合要求的信息
    # 把攻击者数据中符合要求的数据全部提取出来了而已
    if restrict_sampling == True:
        D0_subpop = generate_subpopulation(
            D0, categories=categories, target_attributes=target_attributes
        )
        D1_subpop = generate_subpopulation(
            D1, categories=categories, target_attributes=target_attributes
        )
        print(
            f"D0 has {len(D0)} points with {len(D0_subpop)} members from the target subpopulation"
        )
        print(
            f"D1 has {len(D1)} points with {len(D1_subpop)} members from the target subpopulation"
        )

        return D0_mo, D1_mo, D0, D1, Dp, Dtest, D0_subpop, D1_subpop

    return D0_mo, D1_mo, D0, D1, Dp, Dtest


def generate_Dp(
    # 从测试集采样
    test_df,
    # 目标属性名
    categories=["race"],
    # 目标属性值
    target_attributes=[" White"],
    # 要中毒的标签，也就是假标签，比如现在是1，我们就要把实际为0的改成1，这就是中毒数据
    poison_class=1,
    # 采样数
    k=None,
    # 随机数种子
    random_state=21,
):
    """Generate Dp, the poisoned dataset
    ...
    Parameters
    ----------
        test_df : Pandas Dataframe
            The validation set or some dataset that is disjoint from train set,
            but drawn from the same distribution
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        poison_class : int
            The label we want our poisoned examples to have
        k : int
            The number of points in the poison set
        verbose : bool
            If True, reports dataset statistics via a print-out

    ...
    Returns
    -------
        Dp : Pandas Dataframe
            The adversary's poisoned set
        remaining_indices : np.array
            The remaining indices from test_df that we can use to query the target model.
            The indices correspond to points in the subpopulation that are *not* in the
            poisoned set
    """

    # 如果没有提供要采样的大小，那么就报错
    if k is None:
        raise NotImplementedError("Poison set size estimation not implemented")

    # 先从测试集中采样出符合要求的全部数据，再造个副本，因为要修改，不然原始副本也被改了
    subpop = generate_subpopulation(
        test_df, categories=categories, target_attributes=target_attributes
    ).copy()
    # 提取标签名 label或者class
    label = subpop.columns[-1]
    # 随机数种子
    np.random.seed(random_state)

    # 获取到所有标签为非 poison_class的数据，以便更改标签，来造中毒数据
    subpop_without_poison_label = subpop[subpop[label] != poison_class]

    # 生成索引，再打乱，相当于随机采样，用下边代码应该也可以
    # subpop_without_poison_label = subpop_without_poison_label.sample(n=k).reset_index(drop=True)
    # remain_index = subpop.index.difference(subpop_without_poison_label.index)
    # Dtest =  subpop.iloc[remain_index]
    # 不过写了一圈我发现。本质上还是对索引进行操作，所以这个原始操作就挺好，需要随机采样，又需要剩下那部分就可以直接操作索引
    # 但是测试集就不太好划分了。
    all_indices = np.arange(0, len(subpop_without_poison_label), dtype=np.uint64)
    np.random.shuffle(all_indices)

    # 取出来k个值，用来中毒，前边已经随机采样过了。且不考虑溢出的情况，因为中毒率都很低，那么k都特别小。
    # 但是有时候加上子集攻击（占比很小）的话，可能就没那么靠谱了,最好还是做个对比
    # 直接通过索引访问要用iloc
    Dp = subpop_without_poison_label.iloc[all_indices[:k]]

    # 更改标签，也就是中毒
    Dp.loc[:, label] = poison_class
    # 剩余的那些可以用作测试集
    remaining_indices = all_indices[k:]
    # 测试集中没有被中毒那部分被留作测试
    # 注意这里边没有重置索引，然后我又了解了一下，发现索引重置不重置，其实没有太大区别，所以以前重置操作可以不用
    Dtest = subpop_without_poison_label.iloc[remaining_indices]

    # 这里的test数据集其实就是查询集合
    return Dp, Dtest


# 转化数据为one hot类型
def all_dfs_to_one_hot(dataframes, cat_columns=[], class_label=None):
    """Transform multiple dataframes to one-hot concurrently so that
    they maintain consistency in their columns

        ...
        Parameters
        ----------
            dataframes : list
                A list of pandas dataframes to convert to one-hot
            cat_columns : list
                A list containing all the categorical column names for the list of
                dataframes
            class_label : str
                The column label for the training label column

        ...
        Returns
        -------
            dataframes_OH : list
                A list of one-hot encoded dataframes. The output is ordered in the
                same way the list of dataframes was input
    """

    # 生成一个从0开始的长度为传入的列表数目（所有的数据集，可能不包含dp）的列表，
    keys = list(range(len(dataframes)))

    # Make copies of dataframes to not accidentally modify them
    # 创建副本，避免修改影响原始副本
    dataframes = [df.copy() for df in dataframes]
    # 找到所有的连续列方便后续转换
    cont_columns = sorted(
        list(set(dataframes[0].columns).difference(cat_columns + [class_label]))
    )

    # Get dummy variables over union of all columns.
    # Keys keep track of individual dataframes to
    # split later
    # 只把离散的列转换为one hot
    # 在拼接各个列表的时候，对每一个列表赋予一个固定的key，这样后期可以方便继续拆开，现在拼接是统一操作
    temp = pd.get_dummies(pd.concat(dataframes, keys=keys), columns=cat_columns)
    # get_dummies返回的值是true或者false，替换为0，1，inplace表示是否原地修改（true），还是创建副本
    temp.replace({False: 0, True: 1}, inplace=True)
    
    # Normalize continuous values
    # 离散数据除以他们列的最大值，注意，这个最大值是所有数据集合的最大值，而不是单一的一个数据集的。这样更标准
    temp[cont_columns] = temp[cont_columns] / temp[cont_columns].max()

    # 就是前边操作之后，class列不在最后一列，需要如下操作挪到最后一列并更名为label
    if class_label:
        temp["label"] = temp[class_label]
        temp = temp.drop([class_label], axis=1)
        # temp['label'] = temp.pop(class_label)上边的命令可以用这个语句替代
        # 先删除class_label列，再把返回值赋给label列。一样的作用

    # Return the dataframes as one-hot encoded in the same order
    # they were given
    # 根据key值进行拆分。继续返回
    return [temp.xs(i) for i in keys]


# --------------------------------------------------------------------------------------

# 这个最基础的功能就是读取adult数据集，划分成测试集和训练集返回，并且把class转换为0，1标签。也有一些可选分支，
# 但是基础版本的实验没有设置这些
def load_adult(one_hot=True, custom_balance=False, target_class=1, target_ratio=0.3):
    """Load the Adult dataset."""
    # 默认的文件路径，可以修改为其他的
    # 注意！！！！！！！！！！！！！！！！！！！！！！！！
    # 这个数据集是官方提前已经划分好的，如果想弄可以自己再跑一次。然后我下载官方数据集，虽然一样的东西，但是
    # 大小有点区别我也不知道为什么，数据量是一样的

    filename_train = "dataset/adult.data"
    filename_test = "dataset/adult.test"
    # adult数据集的所有列名,前14个是属性，最后一个是类别
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "class",
    ]
    # 下边这个代码默认的使用的，作为分隔符，但是文件中实际上还有空格，因此这个读取会导致除了第一列之外每一列前边都多一个空格
    # 用如下方法
    # df_tr_1 = pd.read_csv(filename_train, names=names, sep=', ', engine='python')
    # 设置engine为python，不然会警告指定的分隔符（sep=', '）是一个正则表达式（因为它包含了一个空格），
    # 而 'c' 引擎不支持正则表达式的分隔符，所以Pandas 回退到 'python' 引擎。
    df_tr = pd.read_csv(filename_train, names=names)
    # 忽视第一行，因为文件首有一行无关信息
    df_ts = pd.read_csv(filename_test, names=names, skiprows=1)

    # 删除两列，axis为1，表示按列删除，inplace为true表示直接对原表修改
    # 这两个分别是序号和教育信息，共14个属性，去掉两个后变成12个属性
    df_tr.drop(["fnlwgt", "education"], axis=1, inplace=True)
    df_ts.drop(["fnlwgt", "education"], axis=1, inplace=True)

    # Separate Labels from inputs
    # 转换为category类型，category数据类型是pandas中用于表示有限数量的可能值的特殊数据类型
    df_tr["class"] = df_tr["class"].astype("category")
    # 选择类型为category的列，并用columns获得列名，.columns返回的是index对象，可以用于索引DataFrame的列
    # 但是实际上这行代码不必要，因为在adult数据里没有category类型。也就是说cat-Colums只有class列，可以如下替换
    # cat_columns = ['class']或者cat_columns = 'class'
    cat_columns = df_tr.select_dtypes(["category"]).columns
    # apply允许你对DataFrame的每一列或Series的每一个元素应用一个函数
    # 默认的axis为0，在本例中也就是沿着dataframe中的每一列（也就是一个series）处理，之前的axis=0是沿着行
    # lambda函数用法：lambda arguments: expression
    # arguments 是函数参数，可以是多个，用逗号隔开。
    # expression 是一个表达式，它会被求值并返回。
    # .cat.codes应用于series，因为class只有两类，那么实际效果就是把收入小于等于50K的编码为0，大于50K的编码为1
    # 这个代码是可以改变的，如果cat_columns = 'class'，那么df_tr[cat_columns]就是某一列形成的series，那么可以直接应用
    # df_tr[cat_columns] = df_tr[cat_columns].cat.codes
    # 但是如果是cat_columns = ['class']，那么df_tr[cat_columns]还是一个dataframe，是不能直接应用cat.codes的
    # 那么就用原来的代码或者用for循环
    # for col in cat_columns:
    #   df[col] = df[col].cat.codes
    # 下边代码中的x代表df_tr[cat_columns]中的每一列，也就是一个series
    # 注意，转换完之后，category类型会变成int类型
    df_tr[cat_columns] = df_tr[cat_columns].apply(lambda x: x.cat.codes)
    df_ts["class"] = df_ts["class"].astype("category")
    cat_columns = df_ts.select_dtypes(["category"]).columns
    df_ts[cat_columns] = df_ts[cat_columns].apply(lambda x: x.cat.codes)
    # 这里边把连续列和离散列都敲出来了，但是实际上作者已经写过一个这种函数data.get_adult_columns(),
    # 但是虽然总体值是一样的，但是列名的顺序有所不同，暂时未发现这种不同会带来什么影响
    cont_cols = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]  # Indexs: {1,3,5,6,7,8,9,13}
    # pd.concat用于将两个或多个 Pandas 数据结构（如 Series、DataFrame）沿着一条轴（通常是行或列）连接起来
    # 默认是axis=0，也就是沿着行拼接,这里也就是把adult数据集中的测试集和训练集沿着行拼接，还是那么多列，行数变多
    # 合并就是为了处理方便，处理一次就行
    df = pd.concat([df_tr, df_ts])
    # 把所有的离散列转换为category类型，但是没必要循环,下边是一样的.不过现在还不清楚转换之后有什么意义
    # df[cat_cols] = df[cat_cols].astype("category")
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # 这个默认设置为false，也就是没有走这个分支
    if custom_balance == True:
        # 默认的target_class为1，也就是正例，即收入大于50K的，target_ratio为0.3
        # 这个调整的是原始的训练集中你想要的标签样本占比，可能用于做对比实验吧
        df_tr = generate_class_imbalance(
            data=df_tr, target_class=target_class, target_ratio=target_ratio
        )
        # 这个调整的是测试集的
        df_ts = generate_class_imbalance(
            data=df_ts, target_class=target_class, target_ratio=target_ratio
        )

    # 虽然参数默认为true，但是在load data的时候，被设置为false，也就是不需要one hot
    if one_hot == False:
        return df_tr, df_ts

    else:
        # 直接转换为one hot类型，比如A列有两种值，转变后的A列变成两列A_1 和 A_2，都是0或者1
        # get_dummies不会处理int类型，之后处理其他类型
        df_one_hot = pd.get_dummies(df)
        # 把class列改名为label列，注意，在这步之前，class列已经被处理为0，1值了，也就是int类型
        # 为什么非要做这一步操作呢，因为经过上述操作之后，class列不在最后一列了，那么可以生成一个label列然后让他等于
        # class列的值，再把class删掉，就相当于把class重新挪到最后一列
        # 我记得这块跟另一部分代码匹配上了，那边对最后一列做了额外的判定，如果不想额外判定的话，就再操作一次，把label换为class
        df_one_hot["labels"] = df_one_hot["class"]
        df_one_hot = df_one_hot.drop(["class"], axis=1)

        # Normalizing continuous coloumns between 0 and 1
        # df_one_hot[cont_cols].max()返回的是一个series对象，也就是各个数值列的最大值
        # 再用df_one_hot[cont_cols]去除，就是对数值列每一列都除以他们的最大值
        df_one_hot[cont_cols] = df_one_hot[cont_cols] / (df_one_hot[cont_cols].max())
        # 近似为3位小数
        df_one_hot[cont_cols] = df_one_hot[cont_cols].round(3)
        #         df_one_hot.loc[:, df_one_hot.columns != cont_cols] = df_one_hot.loc[:, df_one_hot.columns != cont_cols].astype(int)
        # 重新划分训练集和测试集
        df_tr_one_hot = df_one_hot[: len(df_tr)]
        df_ts_one_hot = df_one_hot[len(df_tr) :]

        return df_tr, df_ts, df_tr_one_hot, df_ts_one_hot


def load_census_data(
    one_hot=True, custom_balance=False, target_class=1, target_ratio=0.1
):
    """Load the data from the census income (KDD) dataset

    ...
    Parameters
    ----------
        one_hot : bool
            Indicates whether one-hot versions of the data should be loaded.
            The one-hot dataframes also have normalized continuous values

    Returns
    -------
        dataframes : tuple
            A tuple of dataframes that contain the census income dataset.
            They are in the following order [train, test, one-hot train, one-hot test]
    """

    filename_train = "dataset/census-income.data"
    filename_test = "dataset/census-income.test"

    column_names = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-stat",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-stat",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-stat",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-stat",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
    ]

    uncleaned_df_train = pd.read_csv(filename_train, header=None)
    uncleaned_df_test = pd.read_csv(filename_test, header=None)

    mapping = {i: column_names[i] for i in range(len(column_names))}
    mapping[len(column_names)] = "class"
    uncleaned_df_train = uncleaned_df_train.rename(columns=mapping)
    uncleaned_df_test = uncleaned_df_test.rename(columns=mapping)

    cont_columns = [
        "age",
        "wage-per-hour",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "instance-weight",
        "num-persons-worked-for-employer",
        "weeks-worked-in-year",
    ]
    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    encoder = LabelEncoder()
    uncleaned_df_train["class"] = encoder.fit_transform(uncleaned_df_train["class"])

    encoder = LabelEncoder()
    uncleaned_df_test["class"] = encoder.fit_transform(uncleaned_df_test["class"])

    uncleaned_df_train = uncleaned_df_train.drop(
        uncleaned_df_train[uncleaned_df_train["class"] == 2].index
    )
    uncleaned_df_test = uncleaned_df_test.drop(
        uncleaned_df_test[uncleaned_df_test["class"] == 2].index
    )

    if custom_balance == True:
        uncleaned_df_train = generate_class_imbalance(
            data=uncleaned_df_train,
            target_class=target_class,
            target_ratio=target_ratio,
        )
        uncleaned_df_test = generate_class_imbalance(
            data=uncleaned_df_test, target_class=target_class, target_ratio=target_ratio
        )

    if one_hot:
        # Normalize continous values
        uncleaned_df_train[cont_columns] = (
            uncleaned_df_train[cont_columns] / uncleaned_df_train[cont_columns].max()
        )
        uncleaned_df_test[cont_columns] = (
            uncleaned_df_test[cont_columns] / uncleaned_df_test[cont_columns].max()
        )

        uncleaned_df = pd.concat([uncleaned_df_train, uncleaned_df_test])

        dummy_tables = [
            pd.get_dummies(uncleaned_df[column], prefix=column)
            for column in cat_columns
        ]
        dummy_tables.append(uncleaned_df.drop(labels=cat_columns, axis=1))
        one_hot_df = pd.concat(dummy_tables, axis=1)

        one_hot_df["labels"] = one_hot_df["class"]
        one_hot_df = one_hot_df.drop(["class"], axis=1)

        one_hot_df_train = one_hot_df[: len(uncleaned_df_train)]
        one_hot_df_test = one_hot_df[len(uncleaned_df_train) :]

        return uncleaned_df_train, uncleaned_df_test, one_hot_df_train, one_hot_df_test

    return uncleaned_df_train, uncleaned_df_test

# 加载数据集
def load_data(data_string, one_hot=False):
    """Load data given the name of the dataset
    ...
    Parameters
    ----------
        data_string : str
            The string that corresponds to the desired dataset.
            Options are {mnist, fashion, adult, census}
        one_hot : bool
            Indicates whether one-hot versions of the data should be loaded.
            # 如果是one-hot类型，连续数据会被归一化
            The one-hot dataframes also have normalized continuous values
    """
    # 这里边只设置了adult何cencus两个数据集，也就是只公布了一部分代码。如果其他部分实验应该是添加更多的选项的
    # 然后在run_attck中，one hot默认为false
    # 用了.lower()操作，统一转换为小写再比较
    if data_string.lower() == "adult":
        # 用默认设置加载数据集，默认不用one hot
        return load_adult(one_hot)

    elif data_string.lower() == "census":
        return load_census_data(one_hot)

    else:
        print("Enter valid data_string")
