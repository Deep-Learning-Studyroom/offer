# 交叉熵(cross entropy)  
假设有两个概率分布p(真实分布)和q(预测分布)。  
交叉熵可以衡量真实分布p和预测分布q之间的差异。**交叉熵，其用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小。** 交叉熵越低，这个策略就越好，最低的交叉熵也就是使用了真实分布所计算出来的信息熵，因为此时交叉熵 = 信息熵。这也是为什么在机器学习中的分类算法中，我们总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明我们算法所算出的非真实分布越接近真实分布。

交叉熵的公式：  
$$H(p || q) = \sum^{N}_{k=1}{p_klog_2\frac{1}{q_k}}$$

交叉熵是分类问题的主流损失函数。

# 相对熵(relative entropy) 
**我们如何去衡量不同策略之间的差异呢？这就需要用到相对熵，其用来衡量两个取值为正的函数或概率分布之间的差异**，即： 
$$KL(f(x) || g(x)) = \sum_{x \epsilon X}{f(x) * log_2\frac{f(x)}{g(x)}}$$
相对熵和交叉熵之间的关系：  
$$KL(p || q) = \sum^{k=1}_{N}{p_k * log_2 \frac{p_k}{q_k}} = H(p, q) - H(p) = \sum^{k=1}_{N}{p_k * log_2 \frac{1}{q_k}} - \sum^{k=1}_{N}{p_k * log_2 \frac{1}{p_k}}$$  
现在，假设我们想知道某个策略和最优策略之间的差异，我们就可以用相对熵来衡量这两者之间的差异。即，**相对熵 = 某个策略的交叉熵 - 信息熵（根据系统真实分布计算而得的信息熵，为最优策略）。**
相对熵（relative entropy）又称为KL散度（Kullback–Leibler divergence，简称KLD），信息散度（information divergence），信息增益（information gain）。

**注意：**KL散度和逆KL散度(p, q位置互换)是不相等的。非对称性意味着使用KL散度或者逆KL散度作为优化目标，其得到的结果将具有显著差异。例如，用分布q去拟合分布p，选择KL散度，Q会将诸多高概率的峰值糊化。

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/use_q_to_fit_p_by_kl_divergence.jpg) 

如果使用逆KL散度，则会导致q去拟合高概率的单峰。
![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/use_q_to_fit_p_by_reverse_kl_divergence.jpg) 

# JS散度  
**KL散度是不对称的， 而JS散度由KL散度计算而来，是对称的。**  
公式如下：  
$$JS(p || q) = \frac{1}{2}KL(p || \frac{p + q}{2}) + \frac{1}{2}KL(q || \frac{p + q}{2})$$

原始GAN的论文中，使用JS散度度量两个分布之间的距离。**为什么使用JS散度效果不好？**因为训练集的概率分布和生成器隐式定义的概率分布往往只是高维空间的低维流形。例如下图在三维空间中，两个分布均是二维流形，其交集最多为一条线段，以至于在计算JS散度时，忽略重叠部分的问题在维数很高时将更加严重。
![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/20190626133454.jpg) 

