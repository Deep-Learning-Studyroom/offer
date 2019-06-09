

## [leetcode-1 two sum](https://leetcode.com/problems/two-sum/)
解法1：枚举。两层嵌套循环，时间复杂度O(n^2)。   
解法2：使用字典，时间复杂度O(n)。  
```
class Solution:
    def twoSum(self, nums: 'List[int]', target: 'int') -> 'List[int]':
        d = {}
        
        for i, x in enumerate(nums):
            if target-x in d:
                return i, d[target-x]
            else:
                d[x] = i
```

## [leetcode-15 3sum](https://leetcode.com/problems/3sum/)
**硅谷常考题**  
解法1：枚举。三层嵌套循环，时间复杂度O(n^3)。  
**解法2：枚举前两个数，然后在字典里查询第三个数。时间复杂度O(n^2)。**  

```python3
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        
        nums.sort()
        res = set()
        for i, a in enumerate(nums[:-2]):
            if i>=1 and a == nums[i-1]:
                continue
            
            d = {}
            for b in nums[i+1:]:
                if b not in d:
                    d[-a-b] = 1
                else:
                    res.add((a, b, -a-b))
        #return [list(x) for x in res]
        #or
        return list(map(list, res))
        
```
注意：
- list在Python2和Python3里不一样。Python2返回列表，Python3返回迭代器，需要外加一个list()函数才能变成列表。
- 开始排序一下，后面方便跳过相同的数。
- 第一层循环遍历的是nums[:-2],不是nums，否则会报错。


## [leetcode-69 sqrtx](https://leetcode.com/problems/sqrtx/)

解法1：二分法

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        l = 0
        r = x
        while l < r:
            mid = (l + r + 1) // 2
            if mid * mid == x:
                return mid
            elif mid * mid > x:
                r = mid - 1
            else:
                l = mid
        return r
```
注意：由于输出是整数，和一般的二分查找不太一样，还是很容易出错的