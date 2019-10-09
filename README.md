# gbdt_var
GBDT衍生变量及其应用

## 衍生GBDT变量
* get\_gbdt\_path\_var  将GBDT各子树的路径衍生为变量,变量名包含了路径的节点信息,便于回溯
* get\_data\_gbdt  回溯GBDT衍生变量给其他数据集（根据各变量取值判断直接回溯，比sklearn的apply和transform更易推广）


## 规则提取(可用于风控策略或反欺诈的规则)
* get\_head\_rule  打印前n个目标占比最高的规则
* get\_rule\_df  计算所有规则的覆盖率、目标占比,返回包含这些信息的数据集
#### 当使用GBDT去提取规则时，需注意几个参数，这几个参数的控制是会影响提取的变量的相关性的（比如min\_samples\_leaf容易剔除相关性强但是覆盖率低的规则）
1. max\_depth 控制每条规则的最多使用变量个数，即一条规则的条件判断不超过max\_depth个
2. min\_samples\_leaf 控制每条规则的最少样本覆盖率，即一条规则的样本覆盖率不小于min\_samples\_leaf（float）
3. n\_estimators 综合max_depth控制规则个数，即提取的规则不超过n\_estimators*2^(max\_depth)个


## 逻辑回归(传统的GBDT+LR实现)
* get\_lr\_model  训练逻辑回归模型,打印并返回模型的截距项、系数、选择变量
* get\_lr\_proba  计算特定截距项、系数、选择变量下的逻辑回归模型的预测概率值(结果与lr.predict_proba相同)
