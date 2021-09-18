1. 统计学习

   基于数据构建统计模型对数据进行分析的一个学科

   1. 数据：所有可被记录的都是数据（数字 文字 图像 语言）
   2. 模型
   3. 策略
   4. 算法

   统计学习方法

   1. 有监督学习 eg. Y（因变量）
   2. 无监督学习 eg. 聚类分析
   3. 强化学习

   学习步骤

   1. 明确学习模型
   2. 明确评价准则
   3. 训练最优模型

2. 统计学习的分类

   基本分类

   1. 监督学习

      从标注数据中学习预测模型
      1. 输入空间&输出空间

         输入与输出所有可能取值的集合

         每一个具体的输入是一个实例（instance）

         由特征向量（feature vector）表示

         $$ x= (x^{(1)}, \cdots, x^{(p)}) \in \mathbb{R}^p$$

         1. 输入与输出可以看作定义有输入空间上随机变量的取值

         2. 监督学习从训练数据（training data），对测试数据（testing data）进行预测

            $ {(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)}$ $(x_i, y_i)$又称为样本(sample)

         3. 输出变量Y $\large{\{}$ 连续型 回归（regression）

            ​						离散型 分类（classification）

      2. 联合概率分布

         X Y 联合概率分布 P(X, Y)

         认为样本(X~i~, Y~i~)依据P(X, Y) 独立同分布产生

      3. 假设空间

         输入空间到输出空间映射的集合

   2. 无监督学习

      1. 从无标注数据中学习模型的机器学习问题

      2. 学习数据的统计规律或潜在结构 eg. 聚类

         假设X是输入空间，Z是隐式结构空间  $z=g(x)\qquad p(\mathop{z}\limits_{输出}|x)$

   按模型分类

   1. 概率模型与非概率模型

      $P(y|x)$ eg. 朴素贝叶斯

      非概率 y=f(x) eg. 神经网络

   2. 线性模型与非线性模型

      y=f(x) $\large{\{}$ 线性函数

      ​				非线性函数

   3. 参数模型与非参数模型

      模型可以由有限维参数刻画

3. 统计学习方法的三要素

   0. 数据

   1. 模型

   2. 策略

      指按照何种准则选择最优模型

      1. 损失函数和风险函数

         损失：度量一次预测的好坏

         风险：度量平均意义模型预测的好坏

         输入变量X

         模型的预测f(x)

         真实值Y

         L(Y, f(x))度量预测错误的程度

         常见的L(Y, f(x))

         1. 0-1 损失函数

            $L(Y, f(x))=\begin{equation}\begin{cases}1 \qquad Y\ne f(x) \\0 \qquad Y=f(x)\end{cases}\end{equation}$ 用于度量分类

         2. 平方损失函数（quadratic loss）

            $L(Y, f(x)) = (Y-f(x))^2$

         3. 绝对损失（absolute loss）

            $L(Y, f(x)) = |Y-f(x)|$

         4. 对数损失 或 对数似然

            $L(Y, f(x))=-log P(y|x)$

            风险函数 损失函数的期望

            E~exp~(f)=E~p~{L(Y, f(X))}=$\int L(y, f(x))P(x, y)dxdy$

            给定一个训练集$T-{(x_1,y_1), \cdots, (x_N, y_N)}

            f(x)关于训练集的平均损失  经验风险（empirical risk）

            $R_{emp}=\frac{1}{N}\sum^{N}_{i=1}L(y_i,  f(x_i))$

            $N\rightarrow +\infty$, 经验风险趋于期望风险

      2. 经验风险最小化与结构风险最小化

         - 经验风险最小化 $\mathop{min}\limits_{f\in\mathbb{F}} \frac{1}{N}\sum^{N}_{i=1}L(y_i,  f(x_i))$

           eg. 极大似然估计是经验损失最小化的例子

           损失函数$\rightarrow$对数似然损失函数

           样本量较大时表现较好

         - 结构风险最小化

           $R_{srm}(f)= \frac{1}{N}\sum^{N}_{i=1}L(y_i,  f(x_i))+\lambda J(f)$ 正则项

           J(f)是模型的复杂度 J(f)越大模型越复杂

   3. 算法

      指模型的具体求解方法，可归结为最优化问题的本解

4. 模型评估和模型选择

   1. 训练误差&测试误差

      $Y=\hat{f}(x)$ 训练误差 关于训练集的平均损失 

      $R_{emp}(f) = \frac{1}{N}\sum^{N}_{i=1}L(y_i,  \hat{f}(x_i))$

      $e_{test}=\frac{1}{N}\sum^{N}_{i=1}L(y_i,  f(x_i))$

      eg. L取0-1损失，$e_{test}=\frac{1}{N}\sum^{N}_{i=1}I(y_i\ne\hat{f}(x_i))$ （误差率error rate） 

5. 正则化和交叉验证

   1. 正则化

      $min \dfrac{1}{N}\sum\limits_iL(y_i, f(x_i))+\lambda J(f)$

      $\lambda$调节参数(training parameter)

      $\lambda \uparrow$ 选择简单模型

      核心：选择经验风险与复杂度同时较小的模型

   2. 交叉验证(cross validation)

      1. 数据充足

         |      |
         |-----------|
         | 训练集(training set) |
         | 验证集(validation set) 用于模型选择|
         | 测试集(testing set) 用于模型评估|

      2. 数据不充足

         需要重复利用数据

         1. 简单交叉验证 

            训练 测试

         2. S折交叉验证

            用s-1折进行训练

            在s折进行模型测试

         3. 留一交叉验证(leave-one-out cross validation)

            S=N （在数据缺乏时使用）

6. 泛化能力

   1. 泛化误差

      设学习到的模型是P

      $R_{exp}(f)=\int L(y-\hat{f}(x))P(x,y)\mathrm{d}x\mathrm{d}y$

      期望风险

   2. 泛化误差的上届

      1. 是样本量N的函数，$N\uparrow$泛化误差$\rightarrow 0$
      2. 是假设空间容量的函数，容量$\uparrow$，误差$\uparrow$

   【例】（二分类问题的误差上界）

   ​	$T=\{(x_1,y_1), \cdots, (x_N, y_N)\} \quad (x_i, y_i) --iid-->P(x,y)$

   ​	$X\in R^p, Y\in{-1,1}$

   ​	假设空间 $F={f_1,\cdot,f_d}

   ​	损失函数0-1

   ​	$R(f)=E[L(Y,X)]=\dfrac{1}{N}\sum\limits_i L(y_i, f(x_i))$

   【Theorem 1】对于以上二分类问题，当假设空间为有限个函数的集合$F={f_1, \cdots, f_d}$时，至少以概率$1-\delta,\, 0<\delta<1$，以下不等式成立：
   $$
   R(f)\le \hat{R}(f)+\epsilon(d,N,\delta)
   $$
   其中，
   $$
   \epsilon(d,N,\delta)=\{\frac{1}{2N}(\log d+\log\frac{1}{\delta}\}^{\frac{1}{2}}
   $$
   

   

   PROOF

   ​	Hoeffding不等式

   ​	$S_n=\sum\limits^n_{i=1}X_i$，$X_i$独立$X_i\in[a_i,b_i]$，对于$\forall t>0$

   ​	$P(S_n-E(S_n)\ge t)\le exp\left(\dfrac{-2t^2}{\sum\limits_{i=1}^{N}(b_i-a_i)}\right)$

