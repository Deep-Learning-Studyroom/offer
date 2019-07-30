## 经验风险最小化和结构风险最小化

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/risk.jpg) 

## SGD、Momentum、NAG、RMSProp和Adam原理

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/gd.jpg) 

## SVM几何间隔和函数间隔

SVM优化目的是找到几何间隔最大的分离超平面。

几何间隔和函数间隔有什么区别和联系呢？

对于训练样本xi，超平面(w, b)，点距离超平面越远表示预测的置信度越高。另外，真实标签是yi，对于二分类问题，可以去1或-1.
下面的式子反应分类的正确性和置信度，就是**函数距离**。

$$y_i(wx_x + b)$$

对于一个训练集来说，函数距离越小，超平面越好。但是函数距离有一个问题，仅仅增加w和b的值就可以放大距离，这并非我们想要的结果。
因此对上式进行归一化。

$$y_i(\frac{w}{||w||}x_i + \frac{b}{||w||}b_i)$$

这就是**几何距离**。这两个距离都是具有几何意义的定义。