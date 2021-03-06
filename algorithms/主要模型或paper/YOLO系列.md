# Yolo v1

## 定性

​	Yolo v1是一种anchor-free的one-stage目标检测算法，它的特点在于经过一系列卷积提取特征后，在最后一层feature map上对每个点进行预测，这个点的box既可以涵盖全图，也可以比较小。Yolov1在卷积之后直接全连接预测出了所有的东西（坐标、类别、置信度），并把它们都转化为了回归问题，所以速度极快。

## 细节

1. Yolo v1是从GoogLeNet改过来的，由24层卷积加2层全连接组成，它先使用imageNet预训练，随后在后面加上了四层卷积和两层全连接组成了Yolo v1。

2. 与SSD不同，首先Yolo v1是anchor-free的，所以它预测出的东西是直接的坐标x、y、w、h，需要注意的是，x、y是相对于当前grid-cell的偏移，所以是0-1之间。而w、h是相对于整张图的宽高，所以也在0-1之间。另外，Yolo v1对每一个grid cell预测出一种类别C和两个box，每个box包含4个坐标和1个confidence，所以每个grid cell是会出现一种类别的box。因此。对于最后SxS的feature map，每个gird预测B个box，模型会预测出SxSx（Bx5+classes））个值。它分配gt的方式是将gt分配到其中心点所在的grid cell，在测试时，我们要使用score的阈值来得到P-R曲线，Yolo v1会将confidence和类别分数相乘，得到一个muiti score来作为score计算。

3. 在损失函数中，Yolo v1全部使用sum-squared error，并且对w和h计算时是开根号的，这是为了改善同样的偏差对大框和小框影响很大的问题。损失函数分为五个部分：x和y的损失、w和h的损失、置信度C的两种计算、以及类别p（c）的计算。其中，xywh的损失只对存在目标的框计算，并且每个grid中只会使用一个IOU与gt最大的进行计算，置信度C会对有目标和没目标的情况都进行计算，但是对有目标的box权重更大。类别p（c）只对有目标的情况计算。

4. 不足：每个格子只预测一个类别，对于重叠的情况不太robust。并且loss对于大框和小框的惩罚力度还是有区别。

  
  
# Yolo v2 && Yolo 9000

  ## Yolo v2

  Yolo v2的文章顺序比较清晰，很明确地说出了有哪些改进，所以我也按照文章顺序整理。

  1. BN和提高分辨率。加入了BN，这个不多说。然后是在fine-tuning阶段，用10个epoch在ImageNet上训练448x448分辨率，高分辨率将最后检测网络的mAP提升了4个点。
  2. 加入Anchor。在Yolo V1中是anchor-free的，而在二代中Yolo也加入了anchor，所以模型预测出来的数值变成了对anchor的偏移量。另外，Yolo v1中是对每个grid cell预测一种类别，对每个box预测一个置信度。v2中改成了对每个anchor预测一个类别和一个置信度。这里有一个细节，为了使每张图的正中间可以被预测一个box（因为大型目标的中心往往在图像的中间），将图像从448改到了416，这样经过32倍降采样后，最终的feature map大小为13x13，是奇数，所以有唯一中点。
  3. anchor的聚类。Yolo v2中使用kmeans来进行anchor的长宽聚类，这里我没弄明白到底是怎样运作的。因为欧式距离对于大小anchor的度量不一致，所以使用IOU来作为距离值。
  
  4. 多尺度训练。Yolo v2使用了不同尺度的图像进行训练，分别是「320，…, 608」，训练时每10个epoch会随机选择一个尺度进行训练，这样的做法是为了使模型拥有对各个尺度图像的检测能力。这种方法有一定价值，在分割中如果目标太小往往也会先用低分辨率图像进行训练，然后再换高分辨率。
  
     

  ## Yolo 9000

Yolo 9000的训练细节我没有弄清楚，总之它的大思想就是用ImageNet来帮助检测的识别部分。 将COCO和ImageNet合并，如果训练时是COCO数据，那么就对整个Yolo的loss进行训练，如果是ImageNet数据，那么就只训练分类分支。



# Yolo V3

Yolo v3也是直接地指出做了哪些改进。精度上其实它比two-stage的差一截，但确实很快。

改进：

1. 对每个boundingbox的置信度和类别预测使用逻辑回归，目的是为了满足多标签的分类任务。
2. 引入了多尺度策略，这应该算是yolo3的一个大的改进。yolo3在三张feature map上进行预测：

- 普通的网络最后一层feature map
- 用最后一层的头两层feature map上采样，然后和更浅层的feature map进行concat操作，接着用卷积来降采样到和最后一层feature map一样的尺寸。
- 用第二步的方法类似地再做一次，但具体是对第几层做的没有说明，需要看源码。
- 总之这种方法带入了多尺度信息，加强了yolo检测小目标的能力。

Things they did that didn't work:

1. 使用box宽、高乘积的线性激活方式来预测box的x、y坐标，影响了模型的稳定性。
2. 用线性回归代替逻辑回归，使模型mAP下降了数个点。
3. 加入focal loss，使mAP下降了两个点。可能是因为Focal loss主要是惩罚分类中的FP的，而yolo中已经将class预测和object的confidence预测解耦了，所以focal loss就无法发挥了。
4. 像faster r-cnn一样取了两个阈值来选框，0.7和0.3。 大于0.7为positive，小于0.3为negative，中间的不管，效果不佳。

总之，这些常识说明，大多数的tricks往往有其特殊的应用场景，并且发挥作用还是需要经过耐心地fine-tuning



​     

​     



