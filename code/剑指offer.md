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