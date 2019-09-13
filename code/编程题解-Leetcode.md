[TOC]

# 1. Sprial Matrix(Leetcode No.54)

Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

<https://leetcode.com/problems/spiral-matrix/>

**Example**：

```tiki wiki
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
```



基本思路：用矩阵的四个角来控制遍历的坐标，当一轮遍历结束后，四个点分别向中心聚拢一点，然后就会变成一个新的螺旋矩阵问题。AC代码如下，速度36ms，占内存13.3MB：



```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        result = []
        r = len(matrix)
        # 防止list为空，导致下面的matrix[0]越界
        if r == 0:
            return result
        c = len(matrix[0])
        x1, x2, y1, y2 = 0, c-1, 0, r-1
        # 待读矩阵至少要为1x1
        while x1 <= x2 and y1 <= y2:
            # 上边
            for i in range(x1, x2 + 1):
                result.append(matrix[y1][i])
            # 右边
            for i in range(y1 + 1, y2 + 1):
                result.append(matrix[i][x2])
            # 当待输入矩阵为一维时，只需上两步即可遍历完
            if x1 < x2 and y1 < y2:
                # 下边
                for i in range(x2 - 1, x1 - 1, -1):
                    result.append(matrix[y2][i])
                # 左边
                for i in range(y2 -1, y1, -1):
                    result.append(matrix[i][x1])
            x1 += 1
            x2 -= 1
            y1 += 1
            y2 -= 1
        return result
```



其中对于x1, x2, y1, y2四点的思考:

1. 当 x1 < x2 and y1 < y2时， 此时待输入的矩阵的维度为n x n(ps. 以下n > 1)
2. 当x1 < x2 and y1 = y2时， 待输入矩阵的维度为1 x n
3. 当x1 = x2 and y1 < y2时，待输入矩阵的维度为n x 1
4. 当x1 = x2 and y1 = y2时， 待输入矩阵的维度为1 x 1

所以，遍历时，矩阵的上边和右边为一组，下边和左边为一组，只要通过控制四点，并且按顺序遍历即可。

# <a href="https://leetcode.com/problems/add-two-numbers/">Add Two Numbers</a>

这道题只需要建立一个carry位，它一直累加本位的数字，给result的本位上赋值carry % 10， 并且到下一位时使用carry /= 10即可获得进位的数字。 唯一要注意的是在carry /= 10时需要强制转换为int类型。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        carry = 0
        p = l1
        q = l2
        result = ListNode(0)
        r = result
        while p != None or q != None:
            carry = int(carry / 10)
            if p != None:
                carry += p.val
                p = p.next
            if q!= None:
                carry += q.val
                q = q.next
            r.next = ListNode(carry % 10)
            r = r.next
        if int(carry / 10) == 1:
            r.next = ListNode(1)
        return result.next
```

# <a href="https://leetcode.com/problems/longest-substring-without-repeating-characters/">Longers substring without repeating characters</a>

这道题要找到字符串中连续地不重复的子串， 注意，要是连续的。所以设置一个最大长度的标识max_c，每一次遍历字母都更新一遍最大值。并设置一个当前子串的开头索引start，用字典记录字母是否出现，字典每次都保存字母最后一次出现的坐标。当发现重复字母时，将start设为”start"和"上一次出现这个字母的位置+1"之间的最大值。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == "":
            return 0
        if len(s) == 1:
            return 1
        max_c = 0
        d = {}
        start = 0
        for i in range(len(s)):
            if s[i] in d:
                start = max(start, d[s[i]] + 1)
            d[s[i]] = i
            max_c = max(max_c, i - start +1)
        return max_c
```

# <a href="https://leetcode.com/problems/median-of-two-sorted-arrays/">Median of Two Sorted Arrays</a>

给定两个排好序的数组，求这两个数组合并后的平均数，要求时间复杂度在O（log（m+n))。我的思路是先算两个list的总长度，如果总长是奇数，则平均数是中间的那个数，如果总长是偶数，则平均数是中间两个数的平均数，以此得到我们想要寻找的数字在总长中的index。然后设置p1,p2两个指针分别指向两个数组， 对前面求出的index ”i“进行遍历，每次将指针指向的数字较小的那一个指针+1，并设置一个i_v保存当前值。这样可以找到合并数组的第i个数，然后得出平均数。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len1 = len(nums1)
        len2 = len(nums2)
        all_len = len1 + len2
        i = -1
        j = -1
        if all_len % 2 == 0:
            i = int((all_len - 1) / 2)
            j = i + 1
        else:
            i = (all_len - 1) / 2
        p1 = 0
        p2 = 0
        i_v = -1
        for _ in range(int(i+1)):
            if p1 < len1 and p2 < len2:
                if nums1[p1] < nums2[p2]:
                    i_v = nums1[p1]
                    p1 += 1
                    
                else:
                    i_v = nums2[p2]
                    p2 += 1
            elif p1 >= len1:
                i_v = nums2[p2]
                p2 += 1
            else:
                i_v = nums1[p1]
                p1 += 1
        if j == -1: # j=-1，说明总长是奇数，直接返回位置i的数字
            return i_v
        else:
            if p1 >= len1:
                return (nums2[p2] + i_v) / 2
            elif p2 >= len2:
                return (nums1[p1] + i_v) / 2
            else:
                return (min(nums1[p1], nums2[p2]) + i_v) / 2
```



# 322. 硬币问题

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        db = []
        for i in range(0, amount + 1):
            db.append(amount + 1)
        db[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if i >= coin:
                    db[i] = min(db[i], db[i - coin] + 1) 
        if db[amount] > amount:
            result = -1
        else:
            result = db[amount]
        return result
```



# 39 组合总和

```python
class Solution:
    def combinationSum(self, candidates, target):
        candidates = sorted(candidates)
        ans = []
        cur = []
        self.dfs(cur, 0, 0, ans, candidates, target)
        return ans


    def dfs(self, cur, sum, used, ans, candidates, target):
        if sum > target:
            return
        if sum == target:
            ans.append(cur)
            return

        for i in range(used, len(candidates)):
            t = cur[:]
            t.append(candidates[i])
            self.dfs(t, sum + candidates[i], i, ans, candidates, target)
            if sum + candidates[i] > target:
                break
```

