

# [leetcode-1 two sum](https://leetcode.com/problems/two-sum/)
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

# [leetcode-15 3sum](https://leetcode.com/problems/3sum/)
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
- map在Python2和Python3里不一样。Python2返回列表，Python3返回迭代器，需要外加一个list()函数才能变成列表。
- 开始排序一下，后面方便跳过相同的数。
- 第一层循环遍历的是nums[:-2],不是nums，否则会报错。


# [leetcode-69 sqrtx](https://leetcode.com/problems/sqrtx/)

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

# [leetcode-703 Kth largest element in a stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)

解法1：维护前k个最大值的数组，每次对k个数进行排序（快排），新的数如果比最小的大的话，就把最小的删除，新的数进入数组，否者不做任何操作。如果要进入n各数，那么时间复杂度是O(nklogk)。

解法2：优先队列。维护一个Min Heap（小顶堆，最上面的元素永远是最小的）。保证堆的元素个数为k，新来的数如果小于堆顶元素，那么不做任何操作，如果大于堆顶元素，那么堆顶元素删除，新的数加到堆里并重新调整堆(O(klog_2(k)))。时间复杂度是O(n(1 or log_2(k)))，最大时间复杂度是O(nlog_2(k))。

```python3
# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)

import heapq
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.pool = nums
        heapq.heapify(self.pool)
        self.k = k
        while len(self.pool) > k:
            heapq.heappop(self.pool)

    def add(self, val: int) -> int:
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, val)
        elif val > self.pool[0]:
            heapq.heapreplace(self.pool, val)
        return self.pool[0]
```

# [leetcode-239 sliding window maximum](https://leetcode.com/problems/sliding-window-maximum/)

**sliding window是常考题**  
解法1：优先队列。维护一个大顶堆(maxheap):
    - 维护heap(删除离开的元素，添加新元素并调整堆)  O(logk)
    - 查最大的元素(直接返回堆顶元素)               O(1)
    - 总时间复杂度O(nlog(k))  
解法2：双端队列(deque)。时间复杂度O(n)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums: return []
        window, res = [], [] #number in window is index, number in res is value 
        for i, x in enumerate(nums):
            if i >= k and window[0] <= i-k:
                window.pop(0)
            while window and nums[window[-1]] <= x:
                window.pop()
            window.append(i)
            if i >= k-1:
                res.append(nums[window[0]])
        return res             
```

