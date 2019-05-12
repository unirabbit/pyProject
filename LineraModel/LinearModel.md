

线性模型形式比较简单，但是却是很多非线性模型的基础形式。深度神经网络的模型其实也是由若干层基础的感知机构成。线性模型的整个思维导图如下图。


 **1. 线性模型的基本形式**
 给定目标数据(**x**,y)，**x**为n维输入向量，y为输出值（一般为一维）。线性模型试图构造一个针对输入的线性组合的函数，来预测其对应输出值。一般向量形式如下：
  <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3-1.png?raw=true" width=120 height=30 />                                                 
 **w**和**b**分别为模型中的权值和偏置量，通过学习后确定，从而确定线性模型的结构。

  **2. 线性回归**
  线性回归问题就是试图学到一个线性模型尽可能准确地预测新样本的输出值。也就是说，我们希望学习到一个 *f*(.) 的关系，使得 *f*(**x**) 的值尽可能地逼近真实的y值。
  那么确定 **w** 和 **b** 的值就是我们最终学习的目标。而这二者值的确定，关键在于衡量 *f*(**x**)【预测值】和 y【真实值】的差别。而均方误差MSE (Mean Squared Error)是回归任务中最常用的性能度量，因此我们可以通过均方差最小化来计算 **w** 和 **b** 的值：
  <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.2.png?raw=true" width=350 height=150 />     
  均方误差几何意义对应了欧氏距离。基于均方误差最小化来进行模型求解的方法称为最小二乘法(least square method)。在线性回归中，最小二乘法就是找到一条离所有样本的欧式距离之和最近的直线。
  而求解 **w** 和 **b** ，使得预测误差**E**最小化的过程被称为线性回归模型的最小二乘参数估计。将误差**E**分别对 **w** 和 **b** 进行求导：
    <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.4.png?raw=true" width=350 height=150 />    
  导数为零时，即误差**E**的一个最小值点，可得到 **w** 和 **b**  最优解的闭式解[1]。
   <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.5.png?raw=true" width=300 height=160 />    
    <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.6.png?raw=true" width=260 height=100 />    
  事实上，最小误差的求解方法除了求偏导以外，还有正规方程的解法。
  **X**为训练集的特征矩阵，最后一个元素恒置于1，即：
  <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.7.png?raw=true" width=350 height=150 />    
  当矩阵**X**的转置与自身相乘为满秩矩阵或者正定矩阵时，**w**有解：
  <img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.8.png?raw=true" width=160 height=45 />    
  注：对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺
寸和米为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的。

对数线性回归：实际上是试图让模型的预测值逼近y的衍生物。实质上是在求取输入空间到输出空间的非线性函数映射。  

 **3. 对数几率回归（逻辑回归）**
 在分类问题中，需要预测的变量 𝑦 是离散的值，我们将学习一种叫做逻辑回归 (Logistic
Regression) 的算法，这是目前最流行使用最广泛的一种学习算法。
逻辑回归也被称为广义线性回归模型，它与线性回归模型的形式基本上相同，都具有 ax+b，其中a和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将ax+b作为因变量，即y = ax+b，而logistic回归则通过函数g将ax+b对应到一个隐状态p，p = g(ax+b)，然后根据p与1-p的大小决定因变量的值。Sigmoid函数（对数几率函数）是一个较为常用的转换函数：
可变换为：
逻辑回归一般使用交叉熵作为代价函数。关于代价函数的具体细节，请参考[代价函数](http://www.cnblogs.com/Belter/p/6653773.html)，这里只给出交叉熵公式：
上式的代价函数是一个关于参数theta的高阶可导连续函数，运用经典的数值优化算法如梯度下降法，牛顿法等都可以求得其最优解。
 **4.  线性判别分析**
 线性判别分析(linear discriminant analysis，LDA)是一类经典的线性学习方法，属于监督学习，也常常用来作为降维处理。主成分分析（PCA）也是一类非监督学习的方法，也是一类常用的降维方法。
 LDA的基本思想：给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样本的投影点尽可能远离；在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别，如下：
 
<img src="https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/3.8.png?raw=true" width=160 height=45 />   
  [1].在解组件特性相关的方程式时，大多数的时候都要去解偏微分或积分式，才能求得其正确的解。依照求解方法的不同，可以分成以下两类：解析解和数值解。

**解析解**(analytical solution)就是一些严格的公式,给出任意的自变量就可以求出其因变量,也就是问题的解, 他人可以利用这些公式计算各自的问题.  
所谓的解析解是一种包含分式、三角函数、指数、对数甚至无限级数等基本函数的解的形式。  
用来求得解析解的方法称为解析法〈analytic techniques〉，解析法即是常见的微积分技巧，例如分离变量法等。  
解析解为一封闭形式〈closed-form〉的函数，因此对任一独立变量，我们皆可将其带入解析函数求得正确的相应变量。  
因此，解析解也被称为闭式解（closed-form solution）

**数值解**(numerical solution)是采用某种计算方法,如有限元的方法, 数值逼近,插值的方法, 得到的解.别人只能利用数值计算的结果, 而不能随意给出自变量并求出计算值.  
当无法藉由微积分技巧求得解析解时，这时便只能利用数值分析的方式来求得其数值解了。数值方法变成了求解过程重要的媒介。  
在数值分析的过程中，首先会将原方程式加以简化，以利后来的数值分析。  
例如，会先将微分符号改为差分符号等。然后再用传统的代数方法将原方程式改写成另一方便求解的形式。  
这时的求解步骤就是将一独立变量带入，求得相应变量的近似解。  
因此利用此方法所求得的相应变量为一个个分离的数值〈discrete values〉，不似解析解为一连续的分布，而且因为经过上述简化的动作，所以可以想见正确性将不如解析法来的好。
  
![enter image description here](https://github.com/unirabbit/pyProject/blob/master/PicResource/LinearModel/linearModelXmind.jpg?raw=true)


> Written with [StackEdit](https://stackedit.io/).