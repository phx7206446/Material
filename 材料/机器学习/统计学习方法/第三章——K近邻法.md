## 3.1 简介
### 3.1.1 直观理解

$k$近邻概念：
![[Pasted image 20220512105540.png]]

由于$k$近邻没有显示的学习过程，所以被称为Lazy Learning。

- 分类问题：对新的实例，根据与之相邻的$k$个训练实例的类别，通过多数表决等方式进行预测。
- 回归问题：对新的实例，根据与之相邻的$k$个训练实例的标签，通过均值计算进行预测。

![[Pasted image 20220512110640.png]]

### 3.1.2 算法

输入：训练集：
$$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$$
其中，$x_i\in\mathcal{X}\subseteq R^n,y\in\mathcal{Y}=\{c_1,c_2,c_3,...,c_k\}$,实例$x$；
输出：实例$x$所属类别$y$。

(1)根据给定的**距离度量**，计算$x$与$T$中点的距离；
(2)在$T$中找到与$x$最近邻的$k$个点，涵盖这$k$个点的$x$的邻域记作$N_k(x)$。
(3)在$N_k{x}$中根据分类决策规则（如多数表决）决定$x$的类别$y$。
$$y=argmax_{c_j}\sum_{x_i\in N_k(x)}I(y_i=c_j),i=1,2,...,N;j=1,2,...,K$$


### 3.1.3 误差率
- 训练集：
$$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$$
- 类别集合：
$$y=\{c_1,c_2,c_3,...,c_k\}$$
- 若考虑最近邻法：对于新实例$x,x_i$为距离它最近的训练实例，两者所属类别分别记作$a$和$b$。
- 误差率：
$$\begin{aligned}
Err(x,x_i)=P(a\ne b|x,x_i)&=\sum_{j=1}^KP(a=c_j,b\ne c_j|x,x_i)\\
&=\sum_{j=1}^KP(a=c_j|x)P(b\ne c_j|x_i)\\
&=\sum_{j=1}^KP(a=c_j|x)(1-P(b=c_j|x_i))
\end{aligned}
$$
- 当$K\rightarrow \infty$时：
$$lim_{K\rightarrow\infty}P(b=c_j|x_i)=P(a=c_j|x)$$
即$x$对应的类别由$x_i$决定。
- 当$K\rightarrow\infty$ 时：
$$
\begin{aligned}
Err(x,x_i)&=\sum_{j=1}^KP(a=c_j|x)(1-P(b=c_j|x_i))\\
&\rightarrow\sum_{j=1}^KP(a=c_j|x)-\sum_{j=1}^KP^2(a=c_j|x)\\
&=1-\sum_{j=1}^KP^2(a=c_j|x)
\end{aligned}
$$
- 假设$x$的真实类别为$c^*$ ，
$$
c^*=argmax_{c_j\in \mathcal{Y}}P(c_j|x)
$$
那么对应的贝叶斯误差率为：
$$P^*(err|x)=1-P(c^*|x)$$
- 误差率第二项：
$$
\begin{aligned}
\sum_{j=1}^KP^2(a=c_j|x)&=P^2(c^*|x)+\sum_{c_j\ne c^*}P^2(c_j|x)\\
&\ge P^2(c^*|x)+\sum_{c_j\ne c^*}(\frac{1-P(c^*|x)}{K-1})^2\\
&=P^2(c^*|x)+\frac{1-P(c^*|x)^2}{K-1}\\
&=(1-P^*)^2+\frac{(P^*)^2}{K-1}
\end{aligned}
$$

- 误差率：
$$
\begin{aligned}
Err(x,x_i)&\le 1-(1-P^*)^2-\frac{(P^*)^2}{K-1}\\
&=2P^*-\frac{K}{K-1}(P^*)2
\end{aligned}
$$


- 当$P^*$较小时，$Err(x,x_i)$上界近似为$2P^*$，则
$$
P^*\le Err(x,x_i)\le 2P^*
$$
- 对于新实例$x$，推广到$k$近邻算法，当$N\rightarrow \infty$且$K\rightarrow \infty$时，$Err(x)\rightarrow P^*$（即，$K$足够大，但相对于$N$又足够小，在大样本数据上，用$k$近邻算法近似于最优决策）


## 3.2 三要素
### 3.2.1 模型

![[Pasted image 20220512144748.png]]

![[Pasted image 20220512144803.png]]


### 3.2.2 三要素
#### 3.2.2.1 距离度量
![[Pasted image 20220512145026.png]]

- 欧式距离（Euclidean distance）：
$$L_2(x_i,x_j)=(\sum_{I=1}^n\lvert x_i^{(I)}-x_j^{(I)}\rvert^2)^{\frac{1}{2}}$$
- 曼哈顿距离（Manhattan distance）：
$$
L_1(x_i,x_j)=\sum_{I=1}^n\lvert x_i^{(I)}-x_j^{(I)}\rvert
$$
- 切比雪夫距离（Chebyshev distance）：
$$L_{\infty}(x_i,x_j)=max_{I}\lvert x_i^{(I)}-x_j^{(I)}\rvert$$
![[Pasted image 20220512145837.png]]

例子：

![[Pasted image 20220512150344.png]]

![[Pasted image 20220512150501.png]]

![[Pasted image 20220512150515.png]]

#### 3.2.2.2 $k-$值的选择

![[Pasted image 20220512150601.png]]

- 较小的$k$值，学习的近似误差减小，但估计误差增大，敏感性增强，而且模型复杂，容易过拟合。
- 较大的$k$值，减少学习的估计误差，但近似误差增大，而且模型简单

> 注：$k$的取值可通过交叉验证来选择，一般低于训练集样本量的平方根。
> 
> 近似误差，可以理解为对现有训练集的训练误差。更关注于“训练”。 
> 估计误差：可以理解为对测试集的测试误差。更关注于“测试”、“泛化”。


#### 3.2.2.3 分类决策规划

多数表决规则：由输入实例的$k$个邻近的训练实例中的多数类决定输入实例的类。

- 分类函数：
$$f:R^n\rightarrow\{c_1,c_2,...,c_k\}$$
- 0-1损失函数
$$ L(Y,f(X))=\begin{cases} 1, & Y\ne f(X) \\ 0, & Y=f(X) \end{cases}$$
- 误分类概率
$$P(Y\ne f(X))=1-P(Y=f(X))$$
给定实例$x\in \mathcal{X}$相应的$k$邻域$N_k(x)$，类别为$c_j$，误分类率：
$$\frac{1}{k}\sum_{x_i\in N_k(x)}I(y_i\ne c_j)=1-\frac{1}{k}\sum_{x_i\in N_k(x)}I(y_i=c_j)$$
最小化误分类率，等价于：
$$argmax\sum_{x_i\in N_k(x)}I(y_i=c_j)$$



## 3.3 $kd$树
### 3.3.1 什么是$kd$树

![[Pasted image 20220512153000.png]]

- 本质：二叉树，表示对$k$维空间的一个划分
- 构造过程：不断地用垂直与坐标轴的超平面将$k$维空间切分，形成$k$维超矩阵区域
- $kd$树的每一个结点对应于一个$k$维超矩形区域

二维情况：

![[Pasted image 20220512153347.png]]



### 3.3.2 构造$kd$树

输入：$k$维空间数据集：
$$T=\{x_1,x_2,...,x_N\}$$
其中，$x_i=(x_i^{(1)},x_i^{(2)},...,x_i^{(k)})^T$

输出：$kd$树
![[Pasted image 20220512153607.png]]

其中第一步中构造根节点，选择某个特征维度作为坐标轴，在实际情况中，一般会计算所有特征维度的方差，然后选取方差最大的那个特征维度作为坐标轴。


### 3.3.3 例题解说

![[Pasted image 20220512154512.png]]

![[Pasted image 20220512154525.png]]

![[Pasted image 20220512154658.png]]

![[Pasted image 20220512154735.png]]

![[Pasted image 20220512154800.png]]

![[Pasted image 20220512154820.png]]

### 3.3.4 搜索$kd$树
#### 3.3.4.1 最近邻搜索
- 寻找“当前最近点”
寻找最近邻的子结点作为目标点的“当前最近点”

- 回溯
以目标点和“当前最近点”的距离沿树根部进行回溯和迭代

具体算法：

![[Pasted image 20220512155606.png]]


#### 3.3.4.2 例题解说

![[Pasted image 20220512155757.png]]

![[Pasted image 20220512155818.png]]


代码可见 https://blog.csdn.net/Galaxy_yr/article/details/89285069