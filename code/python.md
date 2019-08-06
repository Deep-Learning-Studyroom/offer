# python 基础语法 
- python中的“不等于None”不能写"!=None"，要写"is not None"

**python中is比较的是两个对象的内存地址，而==是调用类的__eq__函数比较的。而且在python里面，只要是None内存地址都是一样的，因此用
is比较最准确。用==的话，会根据具体的__eq__函数确定结果**。比如下面的代码：
```
In [2]: class Student:
   ...:     def __eq__(self, other):
   ...:         return True
   ...:

In [3]: s = Student()

In [4]: s is None
Out[4]: False

In [5]: s == None
Out[5]: True

In [6]: id(s)
Out[6]: 2636099000136

In [7]: id(None)
Out[7]: 1471456400
```

- if a is not None 和 if a的区别？

## 深拷贝和浅拷贝

python的赋值语句 b = a 使对象a有了一个**别名**b。这时对b的操作都会影响到a。
```
In [1]: a = [1, 2, 3]

In [2]: b = a

In [3]: b.append(4)

In [4]: b
Out[4]: [1, 2, 3, 4]

In [5]: a
Out[5]: [1, 2, 3, 4]

In [6]: b[0] = 0

In [7]: b
Out[7]: [0, 2, 3, 4]

In [8]: a
Out[8]: [0, 2, 3, 4]
```


这里的拷贝指的是**被拷贝对象的一个副本**，而不是一个**别名**。

如下面的**b=list(a)或b=copy.copy()是一个浅拷贝**对浅拷贝得到的对象添加或删除元素对原对象不会有影响，改变浅拷贝得到的对象的第0个维度，对原对象也没有影响。

**改变浅拷贝得到的对象第0维度之外的元素，可以改变原对象**

```
In [9]: a = [1, 2, 3]

In [10]: b = list(a) # b = copy.copy()

In [11]: b.append(4)

In [12]: b
Out[12]: [1, 2, 3, 4]

In [13]: a
Out[13]: [1, 2, 3]

In [14]: b[0] = 0

In [15]: b
Out[15]: [0, 2, 3, 4]

In [16]: a
Out[16]: [1, 2, 3]


In [17]: a = [[1,2], [3, 4]]

In [18]: b = list(a)

In [19]: b[0][0] = 5

In [20]: b
Out[20]: [[5, 2], [3, 4]]

In [21]: a
Out[21]: [[5, 2], [3, 4]]
```

**b = copy.deepcopy(a)是一个深拷贝**。深拷贝得到的对象和原对象无关，任意修改都不会影响到原对象。
但是深拷贝占的空间更大，这是其缺点。

## python的输入

[参考网站](https://www.jianshu.com/p/6f14ca3290ee)
