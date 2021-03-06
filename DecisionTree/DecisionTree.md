
决策树是一种基本的回归和分类的方法。决策树学习时，利用训练数据，根据损失函数最小化的原则建立决策树模型；预测时，对于新的数据，利用决策树模型进行分类。决策树的学习通常包括三个步骤：特征选择，决策树的生成和决策树的修剪。决策树的生成对应于模型的局部最优，决策树的剪枝对应于模型的全局最优。决策树的类型一般包括ID3、C4.5、CART算法。

## 一.决策树

**1.决策树的基本流程**

决策树是一种树形结构，一颗决策树包含一个根节点、若干个内部节点和若干个叶节点。其中每个内部节点表示一个属性上的判断，每个分支代表一个判断结果的输出，最后每个叶节点代表一种分类结果。决策树是一种十分常用的分类方法，需要监督学习，监督学习就是给出一堆样本，每个样本都有一组属性和一个分类结果，也就是分类结果已知，那么通过学习这些样本得到一个决策树，这个决策树能够对新的数据给出正确的分类。
决策树学习的本质上是从训练数据集中归纳出一组分类规则。对训练数据有较强的分类能力的决策树可能有多个或者不存在。我们需要选择一个对训练数据有很好的拟合能力，同时对未知的数据又有较强的泛化能力的决策树。
决策树学习一般用损失函数表示训练的目标。决策树学习的损失函数通常为正则化的极大似然函数。

**2.决策树的特征选择**

特征选择在于选取对训练数据具有分类能力的特征，特征的分类的能力的强弱需要一些准则进行判断。通常特征选择的准则是信息增益或信息增益比。我们希望决策树的分支结点所包含的样本尽可能属于同一类别，即结点的纯度越来越高。

* 信息增益
熵：表示随机变量不确定性的度量。
信息熵是度量样本集合纯度最常用的一种指标。假定当前样本集合*D*中第*k*类样本所占比例为$p_{k}$(1,2,3...,n),则*D*的信息熵定义为：
$$Ent(D)=-\sum_{k=1}^{n}p_{k}log_{2}p_{k}$$
Ent(D)的值越小，则D的纯度越高。表示对D分类的不确定性。
由定义可知：$0\leqslant Ent(D)\leq log_{2}n$
假定离散属性a有V个可能的取值，特征a对训练数据集D进行划分的信息增益定义如下：
$$Gain(D,a)=Ent(D)-Ent(D|a)=Ent(D)-\sum_{V}^{V=1}\frac{\left | D^{v} \right |}{\left | D \right |}Ent(D^{v})$$
一般而言，熵Ent(D)和条件熵之差成为互信息。信息增益越大，意味着使用该属性a进行划分获得的“纯度提升”越大。
根据信息增益准则的特征选择方法是：对训练数据集D，计算其每个特征的信息增益，并比较其大小，选择信息增益最大的特征。
ID3算法虽然提出了新思路，但是还是有很多值得改进的地方。　　
a)ID3没有考虑连续特征，比如长度，密度都是连续值，无法在ID3运用。这大大限制了ID3的用途。
b)ID3采用信息增益大的特征优先建立决策树的节点。很快就被人发现，在相同条件下，取值比较多的特征比取值少的特征信息增益大。
c) ID3算法对于缺失值的情况没有做考虑
d) 没有考虑过拟合的问题
* 信息增益比（增益率）
信息增益值的大小是相对于训练数据集而言的，并无绝对意义。信息增益可能对可取值数目较多的属性有偏好，使用信息增益比可以修正此问题。
特征a对训练数据集的信息增益比定义如下：
$$g_{R}(D,a) = \frac{gain(D,a)}{Ent(D)}$$
* 基尼系数
用于CART分类树中的特征选择，同时决定该特征的最优二值切分点。

**3.决策树的生成**

* ID3算法
ID3算法的核心是在决策树各个结点上应用信息增益选择特征，递归地构建决策树。具体方法是：从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点；再对子结点递归地调用以上方法构建决策树；直到所有特征的信息增益均很小或者没有特征可以选择为止。最后得到一个决策树。ID3相当于用极大似然法进行概率模型的选择。

* C4.5算法
C4.5算法相当于对ID3算法进行了改进，在生成决策树的过程中，用信息增益比来选择特征。

* CART算法
在ID3算法中我们使用了信息增益来选择特征，信息增益大的优先选择。在C4.5算法中，采用了信息增益比来选择特征，以减少信息增益容易选择特征值多的特征的问题。但是无论是ID3还是C4.5,都是基于信息论的熵模型的，这里面会涉及大量的对数运算。基尼系数简化模型同时也不至于完全丢失熵模型的优点.CART分类树算法使用基尼系数来代替信息增益比，基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。这和信息增益(比)是相反的。
具体的，在分类问题中，假设有K个类别，第k个类别的概率为$p_{k}$, 则基尼系数的表达式为：

$$Gini(p)=\sum_{K}^{k=1}p_{k}(1-p_{k})=1-\sum_{K}^{k=1}p_{k}^{2}$$
## 二.随机森林

## 三.提升树

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190521215120313.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTIxMTcxNTM=,size_16,color_FFFFFF,t_70)


> Written with [StackEdit](https://stackedit.io/).