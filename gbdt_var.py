import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.stats import ks_2samp
import gc


def get_gbdt_path_var(data, model, y=None, one_hot=True, char_cols=[], precision=6):
    '''
    根据数据集和GBDT模型,输出按照GBDT的路径衍生的变量:
    ------------------------------------------------------------
    入参结果如下:
        data: 需衍生变量的数据集,pd.DataFrame对象,不能有缺失
        model: 训练完的GBDT模型
        y: data的输出空间,如果model是未训练的模型,需传入此参数
        one_hot: data是否经过独热处理
            True,data经过了独热处理,此时需保证data有正确的独热后的变量名(独热的变量的变量名应为var_value),默认
            False,data未经过独热处理,此时会帮助data进行独热处理
        char_cols: 类别变量的变量名列表
            由于类别变量是独热后进模型的,衍生的变量也是按照独热后的变量衍生的,会出现var_value > 0.5的节点,
            传入char_cols则可以将节点改为独热前,即var == value,如果data是未经过独热处理的,自动将节点改为独热前
        precision: 提取每个子树的文字结构时,内部节点判断阈值的精确小数位
    ------------------------------------------------------------
    出参结果如下:
        data_gbdt: 按照GBDT的路径衍生了变量的新数据集
    '''
    # 对数据集独热
    if one_hot is False:
        col_types = data.dtypes
        num_cols = list(col_types[col_types != 'object'].index)
        char_cols = list(col_types[col_types == 'object'].index)

        if char_cols != []:
            data_mapper = data[num_cols].join(pd.get_dummies(data[char_cols]))
        else:
            data_mapper = data[num_cols]
    else:
        data_mapper = data.copy()
    # 取独热后的变量名
    mapper_cols = list(data_mapper.columns)

    # 获取类别变量的路径由独热后到独热前所对应的的字典
    if isinstance(char_cols, list) and char_cols != []:
        col_judge_trans = {}
        for col_old in char_cols:
            for col_new in mapper_cols:
                if col_old in col_new:
                    col_judge_trans['%s > 0.5' % col_new] = '%s == %s' % (col_old, col_new.split('_')[-1])
                    col_judge_trans['%s <= 0.5' % col_new] = '%s != %s' % (col_old, col_new.split('_')[-1])

    # 如果未训练模型,这里训练模型
    if isinstance(y, type(None)) is False:
        model.fit(data_mapper, y)

    # 按照GBDT的路径衍生了变量的新数据集,但里面的值时节点索引
    data_gbdt = pd.DataFrame(model.apply(data_mapper)[:, :, 0])

    # 遍历每棵子树
    for n in range(data_gbdt.shape[1]):
        head_label = []
        node_judge = {}
        path_dict = {}

        # 遍历每颗子树内部情况
        for line in export_graphviz(
            model.estimators_[n, 0], feature_names=mapper_cols, precision=precision).split('\n'):
            # 获取现节点的索引
            if '[label=' in line:
                node = int(line[: line.find('[') - 1])
                # 如果现节点有条件判断，保存条件
                if 'label="friedman_mse' not in line:
                    node_judge[node] = line[line.find('label="') + 7: line.find('\\nfriedman_mse')]

            # 上一个节点到下一个节点的路径
            if '->' in line:
                # 获取上一个节点的索引
                node_before = int(line[: line.find('->') - 1])
                # 获取下一个节点的索引
                if '[' in line:
                    node_after = int(line[line.find('->') + 3: line.find('[') - 1])
                else:
                    node_after = int(line[line.find('->') + 3: line.find(';') - 1])

                # 如果上一个节点第一次分裂（上一个节点的条件判断是>=）
                if node_before not in head_label:
                    head_label.append(node_before)
                    # 从根节点到现节点的路径
                    if node_before == 0:
                        path_dict[node_after] = node_judge[node_before]
                    else:
                        path_dict[node_after] = (path_dict[node_before] + ';' + node_judge[node_before])
                # 如果上一个节点第二次分裂（上一个节点的条件判断是<）
                else:
                    # 从根节点到现节点的路径
                    if node_before == 0:
                        path_dict[node_after] = node_judge[node_before].replace('<=', '>')
                    else:
                        path_dict[node_after] = (path_dict[node_before] + ';' + node_judge[node_before].
                                                 replace('<=', '>'))

        if isinstance(char_cols, list) and char_cols != []:
            # 遍历刚才生成的每个节点对应路径的字典
            for index in path_dict.keys():
                # 由于分类变量会进行独热,导致节点的判断为<=0.5/>0.5,将其替换出未独热前的表达方式var ==/!= value
                for old_judge, new_judge in col_judge_trans.items():
                    path_dict[index] = path_dict[index].replace(old_judge, new_judge)

        # 将衍生变量的节点索引转换成路径
        data_gbdt[n] = data_gbdt[n].map(path_dict)

    # 对路径独热
    data_gbdt = pd.get_dummies(data_gbdt, prefix='', prefix_sep='')

    # 去重
    index_tmp = []
    value_tmp = []
    for index, value in enumerate(list(data_gbdt.columns)):
        if value not in value_tmp:
            value_tmp.append(value)
            index_tmp.append(index)

    data_gbdt = data_gbdt.iloc[:, index_tmp]

    return data_gbdt


def get_data_gbdt(data, gbdt_cols):
    '''
    根据get_gbdt_path_var返回数据集的变量列表,回溯gbdt衍生给其他数据集:
    ------------------------------------------------------------
    入参结果如下:
        data: 需衍生变量的数据集,pd.DataFrame对象,不能有缺失
        gbdt_cols: get_gbdt_path_var返回数据集的变量列表
    ------------------------------------------------------------
    出参结果如下:
        data_gbdt: 按照GBDT的路径衍生了变量的新数据集
    '''
    data_gbdt = pd.DataFrame(index=data.index)

    # 变量所有gbdt衍生变量
    for col in gbdt_cols:
        # 需执行exec的字符串初始
        exec_str = "data_gbdt['%s'] = " % col
        judge_len = len(col.split(';'))

        # 遍历变量名中的所有条件加到exec字符串中
        for index, judge in enumerate(col.split(';')):
            # 数值型的条件
            if ('!=' not in judge) and ('==' not in judge):
                if (index == 0) and (index != judge_len - 1):
                    exec_str += ("((data['%s']%s)" % (judge[: judge.find(' ')], judge[judge.find(' '):]))
                elif (index == 0) and (index == judge_len - 1):
                    exec_str += ("(data['%s']%s)" % (judge[: judge.find(' ')], judge[judge.find(' '):]))
                elif index == judge_len - 1:
                    exec_str += (" & (data['%s']%s))" % (judge[: judge.find(' ')], judge[judge.find(' '):]))
                else:
                    exec_str += (" & (data['%s']%s)" % (judge[: judge.find(' ')], judge[judge.find(' '):]))
            # 字符型的条件
            else:
                if '==' in judge:
                    offset = 3
                else:
                    offset = 2

                if (index == 0) and (index != judge_len - 1):
                    exec_str += ("((data['%s']%s'%s')" % (
                        judge[: judge.find(' ')], judge[judge.find(' '): judge.find('=') + offset],
                        judge[judge.find('=') + offset:]))
                elif (index == 0) and (index == judge_len - 1):
                    exec_str += ("(data['%s']%s'%s')" % (
                        judge[: judge.find(' ')], judge[judge.find(' '): judge.find('=') + offset],
                        judge[judge.find('=') + offset:]))
                elif index == judge_len - 1:
                    exec_str += (" & (data['%s']%s'%s'))" % (
                        judge[: judge.find(' ')], judge[judge.find(' '): judge.find('=') + offset],
                        judge[judge.find('=') + offset:]))
                else:
                    exec_str += (" & (data['%s']%s'%s')" % (
                        judge[: judge.find(' ')], judge[judge.find(' '): judge.find('=') + offset],
                        judge[judge.find('=') + offset:]))

        # 将True/False替换成1/0
        exec_str += ".apply(lambda x:1 if x is True else 0)"

        exec(exec_str)

    return data_gbdt


def get_head_rule(X_train_gbdt, Y_train, head=5, cover=0):
    '''
    查看前head个目标占比最高的规则:
    ------------------------------------------------------------
    入参结果如下:
        X_train_gbdt: 衍生了GBDT变量的数据集
        Y_train: 数据集的输出空间
        head: 查看前几个规则
        cover: 覆盖率的阈值,筛选覆盖率大于这个阈值的规则
    '''
    rule_df = pd.DataFrame(columns=['rule', 'cover', 'target'], index=range(X_train_gbdt.shape[1]))
    rule_df['rule'] = X_train_gbdt.columns
    data_count = len(Y_train)
    rule_df['cover'] = [round((X_train_gbdt[rule] == 1).sum() / data_count, 4)
                        for rule in X_train_gbdt.columns]
    rule_df['target'] = [round(Y_train[X_train_gbdt[rule] == 1].sum() / (X_train_gbdt[rule] == 1).sum(), 4)
                         for rule in X_train_gbdt.columns]

    rule_df = rule_df[rule_df['cover'] >= cover].sort_values('target', ascending=False)
    rule_df.index = range(rule_df.shape[0])

    print('总样本目标为 1 的样本占比为: %s\n' % round(Y_train.sum() / data_count, 4))
    print('根据目标占比,提取前 %s 个规则,具体规则如下:' % head)
    for index in range(head):
        print('>> 规则 %s: 覆盖率 %s  目标占比 %s'
              % (index + 1, rule_df.loc[index, 'cover'], rule_df.loc[index, 'target']))
        print('   %s\n' % rule_df.loc[index, 'rule'])


def get_rule_df(X_train_gbdt, Y_train):
    '''
    计算各个规则的覆盖率、目标占比:
    ------------------------------------------------------------
    入参结果如下:
        X_train_gbdt: 衍生了GBDT变量的数据集
        Y_train: 数据集的输出空间
    ------------------------------------------------------------
    入参结果如下:
        rule_df: 含有各个规则覆盖率信息、目标占比信息的DataFrame
    '''
    rule_df = pd.DataFrame(columns=['rule', 'cover', 'target'], index=range(X_train_gbdt.shape[1]))
    rule_df['rule'] = X_train_gbdt.columns
    data_count = X_train_gbdt.shape[0]
    rule_df['cover'] = [round((X_train_gbdt[rule] == 1).sum() / data_count, 4)
                        for rule in X_train_gbdt.columns]
    rule_df['target'] = [round(Y_train[X_train_gbdt[rule] == 1].sum() / (X_train_gbdt[rule] == 1).sum(), 4)
                         for rule in X_train_gbdt.columns]

    rule_df = rule_df.sort_values('target', ascending=False)
    rule_df.index = range(X_train_gbdt.shape[1])

    return rule_df


def get_lr_model(X_train_gbdt, Y_train, C, random_state=1234):
    '''
    根据正则项系数,训练逻辑回归模型:
    ------------------------------------------------------------
    入参结果如下:
        X_train_gbdt: 衍生了GBDT变量的数据集
        Y_train: 数据集的输出空间
        C: L1正则项系数
        random_state: 随机种子
    ------------------------------------------------------------
    入参结果如下:
        intercept: 逻辑回归模型的截距项
        coef: 逻辑回归模型各变量的系数
        cols: 逻辑回归模型选择的变量
    '''
    # 训练模型
    lr = LogisticRegression(penalty='l1', C=C, random_state=random_state)
    lr.fit(X_train_gbdt, Y_train)

    # 提取截距项、系数、选择的变量
    intercept = lr.intercept_[0]
    coef = list(lr.coef_[0][lr.coef_[0] != 0])
    cols = list(X_train_gbdt.columns[lr.coef_[0] != 0])

    # 交叉验证
    valid_auc_list = []
    train_auc_list = []
    fold_5 = KFold(n_splits=5, shuffle=True, random_state=1234)
    for train_index, valid_index in fold_5.split(X_train_gbdt, Y_train):
        X_train_fold, Y_train_fold = X_train_gbdt.iloc[train_index], Y_train[train_index]
        X_valid_fold, Y_valid_fold = X_train_gbdt.iloc[valid_index], Y_train[valid_index]

        lr.fit(X_train_fold, Y_train_fold)
        Y_valid_fold_proba = lr.predict_proba(X_valid_fold)[:, 1]
        Y_train_fold_proba = lr.predict_proba(X_train_fold)[:, 1]

        valid_auc_list.append(metrics.roc_auc_score(Y_valid_fold, Y_valid_fold_proba))
        train_auc_list.append(metrics.roc_auc_score(Y_train_fold, Y_train_fold_proba))

        gc.enable()
        del X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold, Y_valid_fold_proba, Y_train_fold_proba
        gc.collect()

    valid_auc = np.mean(valid_auc_list)
    train_auc = np.mean(train_auc_list)

    print('在L1正则项系数为 %s 下,训练出的逻辑回归模型共选择了 %s 个变量' % (C, len(cols)))
    print('交叉验证结果为:  训练集平均AUC: %s  验证集平均AUC: %s\n' % (round(train_auc, 4), round(valid_auc, 4)))

    print('逻辑回归模型如下:')
    print(('1 / (1 + e^(%s - ' % -intercept + ' - '.join(['%s * X[%s]' % (coef[index], index + 1) for index
                                                          in range(len(coef))]) + '))\n').replace('- -', '+ '))
    print('各变量对应关系如下')
    for index in range(len(coef)):
        print('X[%s] ==> %s' % (index + 1, cols[index]))

    return intercept, coef, cols


def get_lr_proba(intercept, coef, cols, x, y=None):
    '''
    计算特定逻辑回归模型下的预测概率值:
    ------------------------------------------------------------
    入参结果如下:
        intercept: 逻辑回归模型的截距项
        coef: 逻辑回归模型各变量的系数
        cols: 逻辑回归模型选择的变量
        x: 衍生了GBDT变量的数据集
        y: 数据集的输出空间,默认None,如果传入数据集则会打印AUC和KS
    ------------------------------------------------------------
    出参结果如下:
        y_proba: 预测概率值
    '''
    y_proba = x[cols].apply(lambda x: 1 / (1 + pow(np.e, (-intercept - (np.array(x) * np.array(coef)).sum()))), axis=1)

    if isinstance(y, type(None)) is False:
        get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
        auc = metrics.roc_auc_score(y, y_proba)
        ks = get_ks(y_proba, y)

        print('AUC: %s  KS: %s' % (round(auc, 4), round(ks, 4)))

    return y_proba
