# 实现一个卷积函数

```python
import numpy as np

def convolution_without_padding(img_in, kernal):
    (h, w) = img_in.shape
    k = kernal.shape[0]

    img_out = np.zeros([h-k+1, w-k+1])
    for i in range(h-k+1):
        for j in range(w-k+1):
            temp = img_in[i:i+k, j:j+k]
            img_out[i,j] = np.sum(temp * kernal)
    return img_out

def convolution_with_padding(img_in, kernal, padding=0):
    (h, w) = img_in.shape
    k = kernal.shape[0]
    p = padding
    img_in_padded = np.zeros([h+2*p, w+2*p])
    img_in_padded[p:-1*p, p:-1*p] = img_in
    return convolution_without_padding(img_in_padded, kernal)

if __name__ == "__main__":
    img_in = np.array([[40, 24, 135], [200, 239, 238], [90, 34, 94]])
    kernal = np.array([[0, 0.6], [0.1, 0.3]])
    img_out = convolution_without_padding(img_in, kernal)
    print(img_out)
    img_out_2 = convolution_with_padding(img_in, kernal, padding=1)
    print(img_out_2)
```


# 实现一个队图片进行卷积的程序，数据由控制台读入

```python
#数据读取
[h, w] = list(map(int, input().split()))
img_in = []
for _ in range(h):
    img_in.extend(list(map(float, input().split())))
m = int(input())
kernel = []
for _ in range(m):
    kernel.extend(list(map(float, input().split())))
#print(h, w, img_in, m, kernel)
img_out = []


# conv
def get_patch(img_in, i, j, m):
    res = []
    for ii in range(i, i+m):
        for jj in range(j, j+m):
            index = ii * w + jj
            res.append(img_in[index])
    return res


def compute(patch, kernal):
    res = 0
    for val1, val2 in zip(patch, kernal):
        res += val1 * val2
    return res

#
if m == h and m == w:
    img_out = compute(img_in, kernel)
    print(int(img_out))
    exit()

for i in range(w-m+1):
    for j in range(h-m+1):
        patch = get_patch(img_in, i, j, m)
        #print(i, j , patch)
        img_out.append(compute(patch, kernel))

img_out = [int(val) for val in img_out]
#print(img_out)

# print
index2 = 0
for val in img_out:
    if index2 % w == w - m:
        print(val)
        index2 += 1
    else:
        print(val, end=" ")
        index2 += 1

```

>输入数据:
```
3 3
40 24 135
200 239 238
90 34 94
2
0.0 0.6
0.1 0.3
```
>输出结果（向下取整）：
```
106 176
162 174

```