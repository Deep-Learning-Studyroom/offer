# 交叉熵(cross entropy)  
假设有两个概率分布p(真实分布)和q(预测分布)。  
交叉熵可以衡量真实分布p和预测分布q之间的差异。**交叉熵，其用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小。**交叉熵越低，这个策略就越好，最低的交叉熵也就是使用了真实分布所计算出来的信息熵，因为此时交叉熵 = 信息熵。这也是为什么在机器学习中的分类算法中，我们总是最小化交叉熵，因为交叉熵越低，就证明由算法所产生的策略最接近最优策略，也间接证明我们算法所算出的非真实分布越接近真实分布。

交叉熵的公式：  
$$cross \ entropy = \sum^{N}_{k=1}{p_klog_2\frac{1}{q_k}}$$  

# 相对熵(relative entropy)  
相对熵（relative entropy）又称为KL散度（Kullback–Leibler divergence，简称KLD），信息散度（information divergence），信息增益（information gain）。
# KL散度(Kullback–Leibler divergence)  

# JS散度  

