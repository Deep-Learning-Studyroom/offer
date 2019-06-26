# GAN基本原理和常见问题
[一个重要的PPT](http://ice.dlut.edu.cn/valse2018/ppt/Generative_Adversarial_Nets_JSFeng.pdf)
## GAN基本原理
[goodfellow论文]()

对GAN的理解不能停留在简单的“生成器产生样本，判别器分辨样本真假”的阶段。深入的理解是：**先学习一个关于生成器定义的隐式概率分布和训练数据定义的本质概率分布之间的距离度量，然后优化生成器来缩减这个距离度量**。


## GAN的优点
- 计算梯度时只用到了反向传播，而不需要马尔科夫链
- 训练时不需要对隐变量判断
- 理论上，任何可微分函数都能用于构建D和G
- 统计角度上来看，G的参数更新不是直接来源于数据样本，而是使用来自D的反传梯度
- 
## GAN的缺点和常见的问题

- 生成器Pg(G)没有显示的表达
- 比较难训练，D和G之间需要很好的同步，例如D更新k次而G更新1次。下面详细说明GAN为什么训练困难或者训练困难的表现。
  
Goodfellow等从理论上证明了当GAN模型收敛时，生成数据具有和真实数据相同的分布。但是在实践中，**GAN的收敛性和生成数据的多样性通常难以保证**。

主要存在两个问题：生成器梯度消失和模式坍塌（Mode collapse）。

- 梯度消失是指由于生成器G和判别器D的训练不平衡，判别器D的性能很好，对生成数据G(z)和真实数据x能够做出完美的分类，那么D对应的误差损失将很小，进而反向传播到G的梯度值很小，使得生成器G不能得到有效的训练。
- 模式坍塌是指对于任意随机变量z，生成器G仅能拟合真实数据分布pdata的部分模式，虽然G(z)与真实数据x在判别器D难以区分，但是生成器G无法生成丰富多样的数据。

## GAN的损失函数

**d_loss = d_loss_real + d_loss_fake**
- d_loss_real = criterion(outputs, real_labels)
- d_loss_fake = criterion(outputs, fake_labels)

**g_loss = criterion(D(fake_images), real_labels)**

