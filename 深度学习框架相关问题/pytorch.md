# Tensor相关

- 整数型的tensor不能乘以小数
```python
import torch
a = torch.tensor([100, 200])
b = a * 1.2
print(b)
#tensor([100, 200])
```
不过，tensor的默认类型是torch.FloatTensor，因此没有问题。torch.ones生成的也是浮点数1.，而不是整数型的1。
