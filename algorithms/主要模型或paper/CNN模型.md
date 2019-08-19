# [CNN网络结构的发展：从LeNet到EfficientNet](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247489858&idx=1&sn=e4411a314f3001e490cc8933363f37eb&chksm=f9a26bcdced5e2dbc74905ef207f93b7956548e7cb5e22ad843b689a7d2262ed696841d41aca&mpshare=1&scene=1&srcid=&key=4af267131119af88c83424eff678b3d5ae9197400302caa936d4eb44d2184ecf9a676d808ae82f1dcbd0d5b31fd319d6a0f9e554c84693db43bbfe28fe8c265dfa5154e58856c0d213f8f84e7c6e3ea5&ascene=1&uin=MTg4MTg1MDQ4NA%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=T8aYnQfOGRsIfF8maeg%2FlzAkaeuBj1l8KxDdqP7LgcVemVi34vr%2FTn1mSa6M%2B5gU)

# 卷积相关的计算
## 标准卷积的计算

假设输入的图片为[c, h, w]，其中c，h，w分别代表通道数，高和宽。用h_out，w_out，c_out代表输出图片的高、宽、通道。k代表卷积核尺寸，卷积核的数量就是c_out。

$$h_{out} = \frac{h + 2p - k}{s} + 1$$

## 空洞卷积卷积核的实际大小
一个标准卷积核，尺寸为k，空洞系数是d，那么实际卷积核大小为：

$$k' = k + (k - 1) * (d - 1) = d * (k - 1) + 1$$

## 因此空洞卷积输出图片的尺寸为

$$h_{out} = \frac{h + 2p - k'}{s} + 1 = \frac{h + 2p - d(k - 1) - 1}{s} + 1$$

# Inception v1--v4

## Inception v1

Inception v1中的inception模块在一层里使用1x1, 3x3, 5x5的卷积和3x3的maxpooling，然后concatenate一起，好处是**1)增加网络宽度，提高特征表达能力；2)增加了网络对尺度的适应能力，相当于一种多尺度方法**。

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/inception_v1_fig2.png) 

上图中的(b)增加了1x1卷积，减少网络的参数量。**1x1的卷积是一个非常有用的结构，可以跨通道组织信息，提高网络的表达能力，还可以进行输出通道的升维和降维。**

Inception V1有22层深，处理最后一层的输出，中间节点的输出分类效果也很好。因此**在Inception V1中还用到了辅助分类节点：将中间某一层的输出用作分类，并按一个较小的权重（0.3）加载到最终的分类结果中，相当于模型融合**。

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/inception_v1_fig3.jpg) 

## Inception v2

Inception v2改进点：
- 加入了BN层，减少了internal covariate shift（内部神经元的数据分布变化），使每一层的输出都规范化到一个N(0,1)的高斯分布；
- 参考VGG用2个3x3代替inception模块中的5x5，既降低了参数量，也加速计算

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/inception_v2.jpg) 

## Inception v3

v3的主要改进点是**分解（Factorization）**，将7x7分解成两个一维的卷积(1x7, 7x1)，3x3同样(1x3, 3x1)。**好处：减少参数，加速计算(多余的计算力可以用来加深网络)；把一个卷积拆成两个卷积，使得网络深度进一步增加，增加了网络的非线性表达能力。**另外，把输入从224x224变为299x299。

## Inception v4

Inception v4主要研究把inception模块结合残差结构的效果，发现残差结构可以****极大地加速训练，同时性能也有提升**，得到一个inception-resnet-v2网络和一个更深更优化的inception v4模型。

# DenseNet

## 网络结构

**密集连接**。在每一个denseblock中，每一个特征图都会送到后面所有层的输入进行channel-wise的拼接。和resnet不同有两点，一个是每个
residual block内部只有一个跳层连接，二是resnet的跳层连接是element-wise。

**每个block之间是1x1卷积和avgpooling**。因为每个block内部是不会改变

block里面的卷积层是不改变channel的，channel的改变是因为每次都会把之前的拼接起来。block内部特征图size不变，通过block之间的pooling减少size


# ResNeXT

中心思想：Inception那边把ResNet拿来搞了Inception-ResNet，这头ResNet也把Inception拿来搞了一个ResNeXt，
主要就是单路卷积变成多个支路的多路卷积，不过分组很多，结构一致，进行分组卷积。

