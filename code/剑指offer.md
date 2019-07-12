# 面试题3：数组中重复的数字
## 题目一：找出数组中重复的数字。
>在一个长度为n的数组里的所有数字都在0~n-1的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。例如输入长度为7的数组{2, 3, 1, 0, 2, 5, 3}，那么对应的输出是重复的数字2或3.

解法1：哈希表，时间复杂度O(n), 空间复杂度O(n)。
```python 
# input is arr, lenght is n
def find_dupplicat_number(arr):
    n = len(arr)
    d = {}
    for i in arr:
        if i not in d:
            d[i] = 1
        else:
            return i
```
解法2：时间复杂度O(n)，空间复杂度O(1)。充分利用题中任何数都不大于n-1。从前向后遍历数组，如果arr[i] != i，那么比较arr[arr[i]]是否和arr[i]相同，不相同就互换位置，知道arr[i] == i，那么检查后面一个位置的数字。

```python
def find_dupplicat_number(arr):
    n = len(arr)
    for i in range(n):
        while i != arr[i]:
            m = arr[i]
            if m == arr[m]:
                return m
            else:
                arr[i], arr[m] = arr[m], arr[i]
            #print(arr)

print(find_dupplicat_number([2, 3, 1, 0, 5, 3]))
```

## 题目二：不修改数组找出重复的数字。

>在一个长度为n+1的数组里的所有数字都在1~n的范围内，所以数组中至少有一个数字是重复的。请找出数组中任意一个重复的数字，但不能修改输入的数组。例如，如果输入长度为8的数组{2,3,5,4,3,2,6,7}，那么对应的输出是重复的数字2或3。

解法1：同样的，哈希表可以做到时间复杂度O(n)，空间复杂度O(n)。

解法2：不能修改原数组，那么可以复制一个数组，然后在那个数组上利用题目一的解法2，但是时间复杂度O(n)，空间复杂度O(n)。

解法3：这道题和上一道题的区别在于**一定范围内数字的个数不够**。所以可以利用类似二分查找的方法做。

```python
def find_dupplicat_number(arr):
    n = len(arr)
    low = 1
    high = n
    mid = (high + low) // 2

    while(high != low):
        num = 0
        mid = (high + low) // 2
        for i in arr:
            if  low <= i <= mid:
                num += 1
        if num == (mid - low + 1):
            low = mid
        else:
            high = mid
    return(high)

print(find_dupplicat_number([2,3,5,4,3,2,6,7]))
```




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