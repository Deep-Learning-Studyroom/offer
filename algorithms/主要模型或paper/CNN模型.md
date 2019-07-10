# [CNN网络结构的发展：从LeNet到EfficientNet](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247489858&idx=1&sn=e4411a314f3001e490cc8933363f37eb&chksm=f9a26bcdced5e2dbc74905ef207f93b7956548e7cb5e22ad843b689a7d2262ed696841d41aca&mpshare=1&scene=1&srcid=&key=4af267131119af88c83424eff678b3d5ae9197400302caa936d4eb44d2184ecf9a676d808ae82f1dcbd0d5b31fd319d6a0f9e554c84693db43bbfe28fe8c265dfa5154e58856c0d213f8f84e7c6e3ea5&ascene=1&uin=MTg4MTg1MDQ4NA%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=T8aYnQfOGRsIfF8maeg%2FlzAkaeuBj1l8KxDdqP7LgcVemVi34vr%2FTn1mSa6M%2B5gU)

# 标准卷积的计算

假设输入的图片为[c, h, w]，其中c，h，w分别代表通道数，高和宽。用h_out，w_out，c_out代表输出图片的高、宽、通道。k代表卷积核尺寸，卷积核的数量就是c_out。

$$h_{out} = \frac{h + 2p - k}{s} + 1$$

# 空洞卷积卷积核的实际大小
一个标准卷积核，尺寸为k，空洞系数是d，那么实际卷积核大小为：

$$k' = k + (k - 1) * (d - 1) = d * (k - 1) + 1$$

# 因此空洞卷积输出图片的尺寸为

$$h_{out} = \frac{h + 2p - k'}{s} + 1 = \frac{h + 2p - d(k - 1) - 1}{s} + 1$$