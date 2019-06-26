# R-CNN系列文章总结
后面将重新阅读R-CNN系列的文章，从模型的创新、应用两方面进行总结。

<a herf="#R-CNN">R-CNN</a>

<a href="#Fast R-CNN">Fast R-CNN</a>

<a href="#Faster R-CNN">Faster R-CNN</a>

<a href="#Focal loss">Focal loss</a>

<a href="#Mask R-CNN">Mask R-CNN</a>

# R-CNN
## 贡献
R-CNN是第一个在VOC等数据上获得巨大map提升的以CNN为基础的模型（VOC2012上达到了53%），它的主要贡献在于：
1. 在two-stage的cnn base检测算法中，换掉了之前已经使用了至少二十年的sliding-window的提取ROI方式，改用selective-search。因为滑动窗
的窗口固定，实际上限制了感受野，即检测器对目标尺度很敏感，而用ss的方式再resize给网络实际上为模型引入了尺度信息。文章提到用这种方法感受野
提到了195X195，输入是227X227
2. 使用分类问题的预训练模型进行fine-tuning，使map提升了八个点。（解决了标注了box的数据不足的问题）
3. 用回归来修正proposal，提升了3-4个点。
4. 总的来说，提出了一个完整的pipeline，使模型的capacity变大了，然后利用low-level信息来定位，利用high-level信息来分类，也符合浅层位置
信息，深层语义信息的道理。

## 测试过程（forward）
先在ss中得到2000个region proposals，然后将它们resize到固定大小送入cnn，cnn将每个proposal的feature map提取成一个4096维的向量，再通过一个SVM进行分类（SVM的权重矩阵维度是4096 X N， N是类别数目）。
最后对预测出的框做NMS（分别对不同的类别）

## 训练过程
1. 对CNN进行其他数据集的分类预训练。
2. 将预训练好的分类网络拿来用目标检测数据进行分类微调，模型输入是resize后的proposals，输出是（N+1）维，其中N是类别数目，1代表背景。也就是让cnn具有区分N个目标和背景的能力。
label的分配方法是将与gt的iou大于0.5的作正例，其他的都为负例，训练使用SGD，步长为0.001（只有分类预训练时的10%）
3. 上一步训练的batch_size是128.其中分为32个positive（包含了所有类别），96个negative，这是更好锻炼模型区分背景与目标的能力，以提取出有效信息供SVM使用。并且训练时对positive的例子做了一些偏移，为了使这些具有目标的区域也能包含一些背景。
4. 最后训练N个SVM来分类，其中预测的框与某个类别的gt的iou只要大于0.3，就将这个框分配这个类别的label。因为对每个class都训练一个svm很费时间，所以作者选用了对各个class的hard negative的例子来作为训练数据，这样既能提升时间、又能保证性能。

## 疑问
虽然在对cnn进行微调时，有针对背景去分类，但是最后是用的SVM，并没有使用CNN的分类器，那么模型是在哪一步确认这个ROI是不是背景的？ 难道是当每个SVM的输出都是no时，这个框就是背景吗？

## 不足
1. SS的速度很慢，几十秒一张图
2. 对每个类别各训练一个SVM不是很好

# Fast R-CNN
## 主要贡献：
1. R-CNN与SPPnet中，训练很复杂。如先进行fine-tuning训练了一个分类器，却又不用这个分类器最后的softmax部分，而是另外训练SVM，这样就很浪费。并且对回归和分类两个部分是分别训练的，整个pipeline比较复杂。针对这个问题，Fast R-CNN去除了SVM，直接使用fine tuning的softmax作为分类器，并且将回归和分类使用一个multi-loss来训练，这样流程就简洁了许多。
2. 在R-CNN和SPPnet中，模型每得到一个ROI，都要用CNN计算一次，实际上一张图的同一部分可能重复计算了很多次。所以Fast R-CNN将VGG16挪到了流程的最开始，然后引入ROI pool来寻找ROI在卷积后的feature map上对应的位置。通过这种方式，每张图就只需要经过一次backbone了。
## 细节
1. ROIPOOL：
  ROIPOOL是由SPPnet的做法改进而来。在SPPnet中，最后ROI的每一个feature map都被划分成4x4、2x2、1x1三种网格结构，然后对每个网格进行max pooling，再将所有maxpooling的数字堆叠成一维向量。而roipool的做法是将roi投影到卷积后的feature map上，然后将这块区域划分成7x7网格，然后对每个网格做max pool，这样进入后续网络的feature map尺寸就统一了。
2. fine-tuning：将imagenet预训练的模型最后的全连接层移除，换成回归和分类的两个分支，其中分类有K+1个类别，K为类别数。注意网络的输入有两个：图像的list和这些图ROIs的list
3. 模型的训练每个batch采用N=2、R=128，其中N代表图像数、R代表ROI的数量。
4. 回归时，模型得出的是偏移量tx、ty、tw、th，偏移量的形式给模型带来了放缩不变性。
  $$L(p, u, tu, v) = Lcls(p, u) + λ[u ≥ 1]Lloc(tu, v)$$
  其中分类部分是-log，u>=1代表只训练非背景样本。
  并且回归函数中，将偏移量的label：v归一化为0均值、1方差
  。使用smooth函数来做回归，使得反向传播时参数在差值接近0的地方，可以更缓慢地学习，也解决了y=|x|在零点不能求导的问题。
5. 正负例的采样：与gt的iou大于0.5的标正例，（0.1，0.5）区间内的作为负例，原文说这个0.1的阈值是为了挖掘hard example的启发式搜索，所以小于0.1的样本应该是不要了。
6. 分类和回归的分支中，参数的初始化分别使用了标准差为0.01和0.001的零均值高斯分布，bias初始化为0
7. scale invariance： 做了两件事，一件是把每个输入暴力resize成固定尺寸，使模型的输入尺寸一致，学习更稳定。另一件是引入图像金字塔，即把每张图resize成各个尺寸的图，组成金字塔，然后每次随机地抽取一张（受到内存限制），这实际上也是数据增强的一种形式。
8. 对每一类的roi都会做topk的nms。

# Faster R-CNN



## 贡献

主要贡献就是提出了RPN，号称是几乎cost-free的区域建议网络，大大强过Selective Search。

## RPN细节

1. RPN是个全卷积网络，这是为了与faster rcnn共享卷积层。图像经过VGG获得feature map后送入RPN，然后在feature  map的每个像素点上预测K个anchor的信息，其中坐标分支的feature map通道数变成了4K，分类分支变成了2K。这一过程是通过一个3x3卷积，然后每个分支各使用一个1x1卷积来实现的。同时注意因为使用了卷积这种预测方式，给anchor带来了translation invariant（平移不变性）。
2. 训练时分配label，获得positive label的anchor有两种：一种是与某个gt具有最大iou的anchor，另一种是与某个gt的iou超过0.7的anchor，所以一个GT可能会被分配多个anchor。与所有gt的iou都少于0.3的标为negative，其他的anchor不参与训练。损失函数和Fast R-CNN一致。
3. 训练时，每个mini-batch是在一张图上随机采256个anchor进行训练，其中正负比例为1：1。如果这张图的正例不够128，那么就用负例补上。
4. 为什么不能同时训练RPN和检测分支？ 因为检测分支是依赖于修正后的ROI的，并且我们实际上给了检测分支一个先验知识——“ROI里是有目标的”。如果在训练检测分支的同时改变获取ROI的参数，会使这个先验知识变得模糊。所以训练分四步：
   - 先用imagenet预训练的vgg初始化RPN，并训练，并在每张图上收集这个RPN的输出。
   - 再另外用imagenet预训练的vgg初始化一个检测模型，这个模型会直接用到上一步RPN输出的结果，但是并不与RPN共享backbone。
   - 使用第二步训练好的检测模型，再初始化出一个RPN，这个RPN与检测模型共享backbone。然后此时冻结住共享的卷积部分，仅仅只训练RPN的独立部分。
   - 最后一步仅仅微调检测模型的FC部分，其他地方不动。
   - 以上四步很精妙，因为训练的关键是如何让两个部分共享这个VGG网络。VGG网络必须专注于深层次语义的提取，这就必须用于分类的训练。如果VGG被直接用来微调在RPN上，那么imagenet的预训练可能就白费了，因为RPN的重点在浅层的位置信息。但是又不能直接去训练后面检测模型的分类部分，因为检测模型的训练必须要RPN提供的proposals。所以才使用了这种方法，先训练一个临时的RPN，用来提供训练检测模型需要的东西，然后再在这个检测模型的基础上把真正的RPN部分训练出来。
5. 为了减少冗余，两个anchor的IOU如果超过0.7，就要去除一个，去留的条件根据它们的分类 score来决定（NMS）。做完NMS之后，再根据score选取top-k个anchor。

# Focal loss

直入主题，Focal loss这篇文章提出了focal loss，并做了一个retinanet。应该是因为one-stage模型盛行，速度确实很快，所以希望能做一个速度精度都高的模型。

## Focal loss

focal loss的公式是$$FL(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$$, 其中$$p_t= \begin{cases} p  &if \quad y=1\\1-p &otherwise \end{cases}$$ 。先不看$$\alpha_t$$,可以看出，focal loss实际上是加强了对分类置信度较低时的惩罚，无论样本的正负。而因为本文初衷是为了hard negative mining，从而解决easy examples dominant的问题，所以又加上了一个$$\alpha_t$$，它的取值取决于一个作者设定的值$$\alpha$$，当y=1时，$$\alpha_t=\alpha$$；当y=0时，$$\alpha_t=1-\alpha$$。文章中将这个值设为了0.25，也就是说，对于正例有0.25的权重，对于负例有0.75的权重，这样就可以将模型的训练重心放到hard negative examples上来。文章的$$\gamma$$取到2时是最好效果。

## RetinaNet

RetinaNet比较简单，并没有做太大的改动。首先它是one-stage的模型，使用了FPN作为backbone，有预设anchor尺寸，仍然是feature map的每个点有9个anchor。然后对每个feature map分别使用了两个分支预测出坐标偏移量和类别（文章中的类别数是不包括背景的），使用的是3x3卷积组成的FCN。关于NMS，这里用的阈值是0.5，与faster rcnn中的0.7不同。训练时，正例的阈值是0.5，负例的阈值是0-0.4。也就是说0.4-0.5之间的不参与训练。

# Mask R-CNN

Mask R-CNN相较于Faster R-CNN有三个变化： 将backbone改成了FPN，加入了mask分支，将ROIPOOL换成了ROIALIGN。

## ROIAlign

ROIPooling首先会将回归预测数的小数量化回整数，然后划分格子（这个过程仍然会量化为整数），也就是说pool的过程并不平均。而ROIAlign放弃了量化回整数的操作，而是使用双线性插值，获得更精细的像素值，并且此时划分的格子也是严格平均的了。<https://zhuanlan.zhihu.com/p/37998710>这里还说的比较清楚。

## Mask分支

mask分支主要有几个地方要注意

1. mask分支的损失L只在positive的ROI下会计算。
2. mask分支的预测是使用的sigmoid，也就是说在当前ROI下，mask分支会对每种类别都预测出一张mask，然后根据分类分支中预测的分类来选择mask。这样能消除mask预测时的内类竞争，让mask分支专注于轮廓信息。

## 训练细节

1. 分配label，与GT的iou大于0.5的为正，其他的为负（每个版本都在变化，说明这个超参数是随着不同数据要调整的）
2. batch_size是2，每张图都会采64个ROI，正负比例选取为1：3