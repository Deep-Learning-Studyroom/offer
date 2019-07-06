# Yolo v1

## 定性

​	Yolo v1是一种anchor-free的one-stage目标检测算法，它的特点在于经过一系列卷积提取特征后，在最后一层feature map上对每个点进行预测，这个点的box既可以涵盖全图，也可以比较小。Yolov1在卷积之后直接全连接预测出了所有的东西（坐标、类别、置信度），并把它们都转化为了回归问题，所以速度极快。

## 细节

1. Yolo v1是从GoogLeNet改过来的，由24层卷积加2层全连接组成，它先使用imageNet预训练，随后在后面加上了四层卷积和两层全连接组成了Yolo v1。

2. 与SSD不同，首先Yolo v1是anchor-free的，所以它预测出的东西是直接的坐标x、y、w、h，需要注意的是，x、y是相对于当前grid-cell的偏移，所以是0-1之间。而w、h是相对于整张图的宽高，所以也在0-1之间。另外，Yolo v1对每一个grid cell预测出一种类别C和两个box，每个box包含4个坐标和1个confidence，所以每个grid cell是会出现一种类别的box。因此。对于最后SxS的feature map，每个gird预测B个box，模型会预测出SxSx（Bx5+classes））个值。它分配gt的方式是将gt分配到其中心点所在的grid cell，在测试时，我们要使用score的阈值来得到P-R曲线，Yolo v1会将confidence和类别分数相乘，得到一个muiti score来作为score计算。

3. 在损失函数中，Yolo v1全部使用sum-squared error，并且对w和h计算时是开根号的，这是为了改善同样的偏差对大框和小框影响很大的问题。损失函数分为五个部分：x和y的损失、w和h的损失、置信度C的两种计算、以及类别p（c）的计算。其中，xywh的损失只对存在目标的框计算，并且每个grid中只会使用一个IOU与gt最大的进行计算，置信度C会对有目标和没目标的情况都进行计算，但是对有目标的box权重更大。类别p（c）只对有目标的情况计算。

4. 不足：每个格子只预测一个类别，对于重叠的情况不太robust。并且loss对于大框和小框的惩罚力度还是有区别。

   



