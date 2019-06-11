# pix2pix, pix2pixHD, vid2vid三个模型的要点

所有图像到图像的转换问题(image2image translation)都可以用GAN来做。针对这个问题，pix2pix, pix2pixHD, vid2vid三篇论文一脉相承而且效果很好。

学习一个GAN模型，主要理解的是生成器、判别器和训练方法。下面我将从这三个方面对这三个模型进行总结。

## pix2pix
这篇文章的目的是构建一个针对所有"image-to-image translation"问题的通用方法。

这是一个conditional-GAN(cGAN)，它和初始GAN的做法不同的地方在于——cGAN是监督学习。
- 输入到D的数据是image和对应的要转化的图片两者的**按通道拼接**。
### G
**生成器的结构是U-net**，和传统的Encoder-Decoder的区别是把前面的feature map按通道拼接到后面对应的feature map上。防止了encodeing过程中的信息丢失。

### D
**判别器的结构是patch-GAN或者说a patch-based fully convolutional network**。把输入的图片分成N x N的patch，对于每一个patch进行分类（真或假），然后对所有的patch的结果求平均。**这么做的目的是**可以对高频信息进行建模。即给予patch的分类相当于把注意力集中学习比较小的patch的结构。

N在很小时也可以保证输出图片的质量。小尺寸patch的好处是参数更少，运行速度更快，而且小patch也能在大尺寸图像中使用。

patchGAN可以被当做一种texture/style loss。

### 训练方法

- 先训练D一个batch_size，然后训练G一个batch_size，如此循环。
- 训练G的时候并不是$minimize log(1-D(x, G(x,z))$，而是$maximize logD(x, G(x,z))$。这和初始的GAN论文不一样。
- 当优化D的时候，将目标函数除以2，这样相对于G，D学习得会慢一点。
- 评价标准：AMT perceptual studies，FCN-score。

# pix2pixHD

pix2pixHD相对pix2pix增加了：
- high-resolution。是通过**multi-scale** generator and discriminator architecture实现的。
- semantic manipulation。使用**目标实例分割**的信息，这样就有了对象操作的功能，比如添加或删除某个对象。针对同一个输入，可以生成不同的输出，这样可以编辑对象的外表。
  
## G 
**a new coarse-to-fine generator**: $G = \{G_1, G_2\}$。其中，$G_1$是global generator network，$G_2$是local enhancer network。

如果目标是2048 x 1024的图片，那么$G_1$操作的是1024 x 512尺度， $G_2$操作的是$G_1$的四倍分辨率，即2048 x 1024。这种coarse-to-fine的结构可以叠加，$G_1$的输出是1024 x 512, $\{G_1, G_2\}$的输出是2048 x 1024, $\{G_1, G_2, G_3\}的输出是4096 x 2048.

## D 
**multi-scale discriminator**

使用3个相同的网络判别器，每个判别器输入不同尺度的图片。对原图片和对应的目标图片组进行2倍、4倍下采样，这样就得到3个尺度的数据。

虽然三个网络用完全相同的结构，但是有着不同的作用。在最粗的尺度下的网络有最大的感受野，它有更加全局的视角因此可以引导生成器生成更加全局连续的图片；在最精的尺度下的网络可以引导生成器生成更加好的细节。因此**使一个低分辨率的模型增加到高分辨率只需要添加一个高精度的生成器，而不是重新训练**。

没有多尺度的判别器，有的生成图片中会有很多重复的模式。

根据这种判别器，学习问题变成了一个多任务学习问题(multi-task learning problem)

## 训练方法

### 损失函数
**损失函数在原GAN损失函数的基础上添加了feature matching loss**。这个loss涉及到perceptual loss（在图像超分辨率和图像风格迁移中有很好的作用）。feature matching loss会使训练过程更稳定，因为生成器不得不生成在不同尺度下的自然特征。  

feature matching loss，具体来说是从D中的不同的层中提取feature map，然后学习匹配真实图像和生成图像的对应的这些feature map，因此这和perceptual loss很相似。  

### 使用实例分割的图片

之前的图片合成只使用语义分割的图片（semantic label maps），本文还使用了实例分割的图片（instance-level semantic label map）。**instance map相对于semantic label maps多提供的最主要信息，是物体的边界**。为了提取这些信息，本文首先计算instance boundary map。然后，对一个像素如果它的object ID和四个邻居都不一样，那么该像素对应的数字为1，否则是0.

D的输入是下面三张图的拼接，instance boundary map、semantic label map和真实或合成的图片。


# vid2vid

任务：从一个视频生成另一个视频。

本文之前有很多image2image的方法，但是直接把图片转换的方法用来转换视频中的每一帧是不行的，会丢掉帧与帧之间的时间上的动态信息。