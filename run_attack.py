import argparse
import numpy as np
from tqdm import tqdm
import warnings
# 忽视所有警告
warnings.filterwarnings("ignore")
from propinf.attack.attack_utils import AttackUtil
import propinf.data.ModifiedDatasets as data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 去掉string中所有在chars中的元素，用于去掉参数中的括号和空格信息
def remove_chars(string, chars):
    out = ""
    for c in string:
        if c not in chars:
            out += c
    return out
# 我自己改进的版本。因为字符串不可扩充每次+都相当于创建新的字符串，但是列表是可扩充的，这样开销小一点
# def my_remove_chars(string, chars):
#     out = [c for c in string if c not in chars]
#     return  ''.join(out)

# 这个是用于处理参数中的中毒率，转为float列表[0.03, 0.05]
def string_to_float_list(string):
    # Remove spaces
    string = remove_chars(string, " ")
    # Remove brackets
    string = string[1:-1]    
    # Split string over commas
    tokens = string.split(",")
    
    out_array = []
    for token in tokens:
        out_array.append(float(token))
    return out_array

# 我自己的版本
# def my_string_to_float_list(string):
#     string = string[1:-1]
#     string = remove_chars(string, " ")
#     out_array = [float(token) for token in string.split(",")]
#     return  out_array

# 处理参数中的属性，例如[(race, White), (sex, Male)],转换为[('race', 'White'), ('sex', 'Male')]
def string_to_tuple_list(string):
    # Remove spaces
    string = remove_chars(string, " ()")
    # print
    # Remove brackets
    string = string[1:-1]    
    # Split string over commas
    tokens = string.split(",")
    
    targets = []
    # 这里的//2+1应该是错的，当参数长度超过4的时候就错了。但是例子都是比较短的所以没问题
    for i in range(0, len(tokens)//2+1, 2):
        targets.append((tokens[i], tokens[i+1]))
    return targets

# def my_string_to_tuple_list_1(string):
#     string = string[2:-2].replace(" ", "")
#     # string = remove_chars(string, " ")
#     pairs = string.split("),(")
#     targets = [(pair.split(",")[0], pair.split(",")[1]) for pair in pairs]
#     return  targets
#
# def my_string_to_tuple_list_2(string):
#     string = string[1:-1]
#     string = remove_chars(string, " ()")
#     tokens = string.split(",")
#     targets = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)]
#     return targets



if __name__ == '__main__':

    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 定义命令行参数 -dat,和--dataset为参数的两种输入形式,help为运行python run_attack.py --help或者-h时输出的信息
    # 定义了参数类型为str,并且参数默认值为adult数据集
    parser.add_argument(
        '-dat',
        '--dataset',
        help='dataset name',
        type=str,
        default='adult'
    )
    
    parser.add_argument(
        '-tp',
        '--targetproperties',
        help='list of categories and target attributes. e.g. [(sex, Female), (occupation, Sales)]',
        type=str,
        default='[(sex, Female), (occupation, Sales)]'
    )

    # 关于参数传递，作者用的=来传递t0，会报错，直接-t0 0.01就行
    parser.add_argument(
        '-t0',
        '--t0frac',
        help='t0 fraction of target property',
        type=float,
        default=0.4
    )
    
    parser.add_argument(
        '-t1',
        '--t1frac',
        help='t1 fraction of target property',
        type=float,
        default=0.6
    )
    
    parser.add_argument(
        '-sm',
        '--shadowmodels',
        help='number of shadow models',
        type=int,
        default=4
    )
    
    parser.add_argument(
        '-p',
        '--poisonlist',
        help='list of poison percent',
        type=str,
        default= '[0.03, 0.05]'
    )
    
    parser.add_argument(
        '-d',
        '--device',
        help='PyTorch device',
        type=str,
        default= 'cpu'
    )
    
    parser.add_argument(
        '-fsub',
        '--flagsub',
        help='set to True if want to use the optimized attack for large propertie',
        type=bool,
        default= False
    )
    
    parser.add_argument(
        '-subcat',
        '--subcategories',
        help='list of sub-catogories and target attributes, e.g. [(marital-status, Never-married)]',
        type=str,
        default='[(marital-status, Never-married)]'
    )
    # 我的修改版本,去掉默认值,这样就可以对这个进行判断了,否则因为默认值的存在if arguments["subcategories"]:总是成立
    # parser.add_argument(
    #     '-subcat',
    #     '--subcategories',
    #     help='list of sub-catogories and target attributes, e.g. [(marital-status, Never-married)]',
    #     type=str
    # )
    
    parser.add_argument(
        '-q',
        '--nqueries',
        help='number of black-box queries',
        type=int,
        default=1000
    )
    
    parser.add_argument(
        '-nt',
        '--ntrials',
        help='number of trials',
        type=int,
        default=1
    )
    # parser.parse_args()为解析实际中的命令行参数
    # vars() 是一个内置函数，它返回一个对象的 __dict__ 属性。这个属性包含了对象的属性及其对应的值
    # 下边arguments就是参数的字典形式
    arguments = vars(parser.parse_args())
    arguments["poisonlist"] = string_to_float_list(arguments["poisonlist"])
    arguments["targetproperties"] = string_to_tuple_list(arguments["targetproperties"])
    # 下边这个应该有点问题,因为有默认值,他一直都是成立的,也就一直会执行子集攻击好像。
    # 但是后边又发现他通过一个参数来控制是否会执行优化版本的子集攻击。这里只是简单的获取参数值
    if arguments["subcategories"]:
        arguments["subcategories"] = string_to_tuple_list(arguments["subcategories"])
        # print(f"running sub attck on {arguments['subcategories']}")
    # 打印攻击开始，及感兴趣的属性
    print("Running SNAP on the Following Target Properties:")
    for i in range(len(arguments["targetproperties"])):
        print(f"{arguments['targetproperties'][i][0]}={arguments['targetproperties'][i][1]}")
    # 效果如下:
    # sex = Female
    # occupation = Sales
    # 这个代码可以改进为如下地方
    # for property_name,property_value in args["targetproperties"]:
    #     print(f"{property_name}={property_value}")
    print("-"*10)

    # 获取adult数据集中的连续项和离散项
    # 这块有点问题,因为他不是根据参数来判断的,而是直接加载的就是adult数据集,可以写一个if判断来判断参数决定加载的数据集
    cat_columns, cont_columns = data.get_adult_columns()
    # 这步就是获取要分析的数据集名字
    # 但是按理说只能是adult,除非上边代码改了
    dataset = arguments["dataset"]
    # 加载adult的训练集和测试集，做了一点处理，转标签为0，1形式
    df_train, df_test = data.load_data(dataset, one_hot=False)

    # 攻击的属性的名称，比如性别这种
    categories = [prop[0] for prop in arguments["targetproperties"]]
    # 攻击的属性的具体值，比如男，女
    target_attributes = [" " + prop[1] for prop in arguments["targetproperties"]]
    # 上边不改的话这个代码应该一直判断为真，这步只是用来获取值而已，是否执行优化版本的攻击要看具体的控制参数
    if arguments["subcategories"]:
        sub_categories = [prop[0] for prop in arguments["subcategories"]]
        # 为什么前边加了空格暂时不明确，
        sub_attributes = [" " + prop[1] for prop in arguments["subcategories"]]
    else:
        sub_categories = None
        sub_attributes = None

    # 读取参数获得两个world的目标属性比例
    t0 = arguments["t0frac"]
    t1 = arguments["t1frac"]

    # 试验几次，默认为1次
    n_trials = arguments["ntrials"]
    # 黑盒查询次数，默认1000，用于模拟分布。相当于用1000个样本去查询目标模型来获得logit分布
    n_queries = arguments["nqueries"]
    # 查询轮数。每一轮都是查询n_queries个样本，每一轮计算一次推断正确的数目，最后统计正确率的均值
    num_query_trials = 10
    # 定义一个中毒率字典
    avg_success = {}
    pois_list = arguments["poisonlist"]
    
    # 初始化属性推断的攻击类
    # target_model_layers是目标模型的结构
    # verbose为true的时候会打印提示信息，默认不打
    attack_util = AttackUtil(
    target_model_layers=[32, 16],
    df_train=df_train,
    df_test=df_test,
    cat_columns=cat_columns,
    verbose=False,
    )
    
    # 开始逐个按照中毒率执行攻击
    for pois_idx, user_percent in enumerate(pois_list):
        # 当前中毒率的成功率初始化为0
        avg_success[user_percent] = 0.0

        # 设置攻击超参数，其他未指明的是默认值
        attack_util.set_attack_hyperparameters(
            # 攻击的属性名，性别
            categories=categories,
            # 攻击的属性值，女
            target_attributes=target_attributes,
            # 子类攻击属性名
            sub_categories=sub_categories,
            # 子类攻击属性值
            sub_attributes=sub_attributes,
            # 是否想使用优化版本的攻击，也就是利用子属性？默认为false也就是不用
            subproperty_sampling=arguments["flagsub"],
            # 当前中毒率
            poison_percent=user_percent,
            # 想要中毒的类别，这里是1也就是收入大于50K的。也可也设置为其他的
            poison_class=1,
            t0=t0,
            t1=t1,
            # 查寻次数
            num_queries=n_queries,
            # 目标模型数量，也就是对每个world训练10个目标模型，以供推断，进而计算出推断准确率
            num_target_models=10,
        )
        # 设置阴影模型的超参数
        attack_util.set_shadow_model_hyperparameters(
            # 默认cpu上训练，也可以设置为gpu
            device=arguments["device"],
            # 工作核数量
            num_workers=1,
            # 模型框架及训练设置
            batch_size=256,
            layer_sizes=[32,16],
            # 是否打印提示信息
            verbose=False,
            # 母鸡
            mini_verbose=False,
            # 训练20轮
            epochs=20,
            tol=1e-6,
        )

        # 默认只跑一次实验，但是前边单独设置了10，也就是跑10次再计算均值
        for i in range(n_trials):

            # 生成数据集
            attack_util.generate_datasets()

            # need_metrics用于指示是否需要返回模型训练的详细指标，如损失值和准确率
            # 但是这个函数根本没实现这个功能
            attack_util.train_and_poison_target(need_metrics=False)

            (
                out_M0,
                out_M1,
                threshold,
                correct_trials,
            ) = attack_util.property_inference_categorical(
                num_shadow_models=arguments["shadowmodels"],
                query_trials=num_query_trials,
            ) # 开始执行攻击

            # 统计当前中毒率下的成功率
            # 因为这里边其实只跑一次，所以avg_success[user_percent] = correct_trials / n_trials就行
            # 如果多次实验，那么是要求均值的
            avg_success[user_percent] = (
                avg_success[user_percent] + correct_trials / n_trials
            )

    # 打印成功率
    print("Attack Accuracy:")
    for key in avg_success:
        print(f"{key*100:.2f}% Poisoning: {avg_success[key]}")
    
    
