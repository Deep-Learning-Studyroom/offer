# Single Shot MultiBox Detector

首先对SSD算法定性：它是一个基于Anchor的one-stage的模型。

## 模型细节

1. SSD的主模型是基于VGG16的，论文的例图中输入是300 * 300的图，在尾端有38 * 38、19 * 19、10 * 10、5 * 5、 3 * 3、 1 * 1六个尺度的feature map会被送入全连接层预测出坐标和score。这里实际上是构建了一个图像金字塔，给模型带来了多尺度策略。
2. SSD是基于anchor的，它预测的坐标是与r-cnn家族一样的偏移量。并且每一个尺度的feature map所预设的Anchor的尺度是不同的，其计算方法是$$s_k=s_{min}+\frac{s_{max}-s{min}}{m-1}(k-1), k\in[1,m]$$, 其中$s_{min}=0.2,s_{max}=0.9$, m是feature map的数量，这样每张feature map都被分配了一个scale。 然后每个feature map预设anchor的长宽是$$w_k^\alpha=s_k\sqrt{\alpha_r}, h_k^\alpha=h_k\sqrt{\alpha_r}, \alpha\in[1,2,3,\frac{1}{2},\frac{1}{3}]$$, $\alpha$代表的是aspect ratio。**这里有一个细节**，虽然$\alpha$只有五个角度变换，但其实对于1来说，还有一个额外的$s_k^`=\sqrt{s_ks_{k+1}}$。这是因为其它四种ratio都是成对的，所以给1这个ratio多添加了一个稍大的anchor。故在feature map上的每个点，都会预测出6组anchor的偏移量，每个anchor的长宽由所在的feature map决定。
3. loss函数与r-cnn的差不多，也是只有positive的box才会参与回归分支的训练，positive和negative的box都会参与cls分支，cls分支是对正确的那一类求负对数损失。
4. 训练时，采取负例：正例为3：1的比例进行采样训练。

## PS. 目标检测的一个细节

1. 检测任务的训练和inference时实际上有一个不一样的细节。训练时，训练样本是根据所有模型预测的box中与GT的IOU阈值来挑选正例和负例（**注意，这里不会管box的score是否大于某阈值**），正例和负例都会送入cls分支进行训练，然后只有正例会送入回归分支训练。所以这时对某一类别是存在TN、TP、FN、FP四个指标的，是T还是F由这个box对该类别的置信度是否大于预设的阈值来决定。
2. 那之前说过检测不用ROC曲线是因为没有TN，但这里不是有吗？ 这是因为在训练过程中，TN首先得是挑选出来的那些Negative的例子，然后当模型确实认为这个例子是属于背景类别是，它就是TN。但是在inference时，我们不会去挑一些框出来评测，对一张图而言，只要没有被预测出框，并确实是背景的地方都是TN，也就是说每个像素点都可能有几个TN，所以TN的获取就很难，故不使用ROC曲线，而是使用不需要TN的P-R曲线。