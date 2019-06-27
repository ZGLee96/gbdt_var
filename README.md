# gbdt_var
GBDT衍生变量及其应用

## 衍生GBDT变量
* get\_gbdt\_path\_var  将GBDT各子树的路径衍生为变量,变量名包含了路径的节点信息,便于回溯
* get\_data\_gbdt  回溯GBDT衍生变量给其他数据集


## 规则提取(可用于风控策略或反欺诈的规则)
* get\_head\_rule  打印前n个目标占比最高的规则
* get\_rule\_df  计算所有规则的覆盖率、目标占比,返回包含这些信息的数据集


## 逻辑回归(传统的GBDT+LR实现)
* get\_lr\_model  训练逻辑回归模型,打印并返回模型的截距项、系数、选择变量
* get\_lr\_proba  计算特定截距项、系数、选择变量下的逻辑回归模型的预测概率值(结果与lr.predict_proba相同)