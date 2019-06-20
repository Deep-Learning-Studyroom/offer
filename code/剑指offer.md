# 斐波那契数列(Fibonacci) 

解法1：递归。指数级的时间复杂度，面试官不会喜欢。

```python
def fibo(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    return fibo(n-1) + fibo(n-2)
```

解法2：递推公式，时间复杂度O(n)  
```python
def fibo(n):
    first_number = 0
    second_number = 1

    if n == 0:
        return 0
    elif n == 1:
        return 1
    for _ in range(n):
        first_number, second_number = second_number, first_number+second_number
    return first_number
```
解法3：矩阵相乘法，时间复杂度O(log(n))  


# 补充题：从无序数组中找到第k大的数

```python
def partition(num, low, high):
    pivot = num[low] 
    while low < high:
        while low < high and pivot < num[high]:
            high -= 1
        while low < high and num[low] < pivot:
            low += 1
        num[low], num[high] = num[high], num[low]
    num[high] = pivot
    return high, num

def findkth(num, low, high, k): # #找到数组里第k大的数，从0开始
    index = (partition(num, low, high))[0]
    #print(partition(num, low, high)[1])
    if index == k:
        return num[index]
    elif index < k:
        return findkth(num, index+1, high, k)
    else:
        return findkth(num, low, index-1, k)
```


# 面试题39：数组中出现次数超过一半的数字

题目：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。  

解法1： 从数组的特点出发，只需要在遍历数组时保存两个数，一个是数组中的一个数，一个是次数。当遍历到下一个数字时，如果次数为0，则把当前的数字赋值给保存的数字，并把次数置为1，如果下一个数字和之前保存的数字相同，则次数加一，否则次数减一。
```python
def more_than_half_num(num):
    result = num[0]
    times = 0
    for i in num:
        if times == 0:
            result = i
            times = 1
        elif result == i:
            times += 1
        else:
            times -= 1
    return result
```

#面试题40：无序数组里最大的k个数
**这个题绝对是高频题，很多面经里面都提到了** 

解法1：当我们可以修改输入的数组时，使用partition函数可以做到时间复杂度为O(n)。

解法2：当处理海量数据时，我们一般不能修改输入的数组，此时可以用**维护一个小顶堆**的方法。时间复杂度为O(nlog(k))。

解法2代码  
```python
from heapq import *
def kth_largest(num, k):
    if len(num) < k:
        raise ValueError
    heap = num[:k]
    heapify(heap) # 默认是小顶堆
    for i in num[k:]:
        if i <= heap[-1]:
            pass
        else:
            heappop(heap)
            heappush(heap, i)
    return heap
```