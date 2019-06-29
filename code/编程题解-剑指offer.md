 # 编程题解-剑指offer
## 数组中的逆序对
```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        length = len(data)
        if length <= 0:
            return 0
        sub_data = []
        for i in range(len(data)):
            sub_data.append(data[i])
        count, _, _ = self.InverseHelper(data, sub_data, 0, length - 1)
        return count % 1000000007

    def InverseHelper(self, data, sub_data, start, end):
        if start == end:
            sub_data[start] = data[end]  # delete
            return 0, data, sub_data
        length = int((end - start) / 2)
        left, sub_data, data = self.InverseHelper(sub_data, data, start, start + length)
        right, sub_data, data = self.InverseHelper(sub_data, data, start + length + 1, end)
        p_left = start + length
        p_right = end
        sub_i = end
        count = 0
        while p_left >= start and p_right >= start + length + 1:
            if data[p_left] > data[p_right]:
                sub_data[sub_i] = data[p_left]
                sub_i -= 1
                p_left -= 1
                count = count + p_right - start - length  # count = count + p_right - (start + length + 1) + 1
            else:
                sub_data[sub_i] = data[p_right]
                sub_i -= 1
                p_right -= 1
        while p_left >= start:
            sub_data[sub_i] = data[p_left]
            sub_i -= 1
            p_left -= 1
        while p_right >= start + length + 1:
            sub_data[sub_i] = data[p_right]
            sub_i -= 1
            p_right -= 1
        return left + right + count, data, sub_data

if __name__ == '__main__':
    a = [364,637,341,406,747,995,234,971,571,219,993,407,416,366,315,301,601,650,418,355,460,505,360,965,516,648,727,667,465,849,455,181,486,149,588,233,144,174,557,67,746,550,474,162,268,142,463,221,882,576,604,739,288,569,256,936,275,401,497,82,935,983,583,523,697,478,147,795,380,973,958,115,773,870,259,655,446,863,735,784,3,671,433,630,425,930,64,266,235,187,284,665,874,80,45,848,38,811,267,575]
    solution = Solution()
    result = solution.InversePairs(a)
    print(result)
```
归并排序思想，这个版本是从讨论区java版本复现来的，但是说时间复杂度过高，很郁闷。总之这道题思路就是在归并排序的过程中统计逆序对，具体见https://www.nowcoder.com/profile/1591420/codeBookDetail?submissionId=15823415

# <a href="https://www.nowcoder.com/practice/f78a359491e64a50bce2d89cff857eb6?tpId=13&tqId=11199&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking">孩子们的游戏</a>

一个环，由n个顺序数字组成（0 ~ n-1），任给一个数字m，从0开始往环的前向走，将第m-1个数字移出队伍。然后从这个数字的后一个数字起再移出它后面的第m-1个数字。直到这个环只剩一个数字时，输出这个数字。

```python
def LastRemaining_Solution(n, m):
  if not n or not m:
    return -1
  kids = range(n)
  start = 0
  while(len(kids) > 1):
    start = (start + m -1) % len(kids)
    kids.pop(start)
  return kids[0]
```



# 二叉搜索树的第K个节点

给一棵二叉搜索树的根节点和k，给出第k大的那个节点。直接使用中序遍历即可，以下是非递归版本

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Stack: 
    """模拟栈""" 
    def __init__(self): 
        self.items = [] 

    def isEmpty(self): 
        return len(self.items)==0  

    def push(self, item): 
        self.items.append(item) 

    def pop(self): 
        return self.items.pop()  

    def peek(self): 
        if not self.isEmpty(): 
            return self.items[len(self.items)-1] 

    def size(self): 
        return len(self.items) 
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        if not pRoot or k == 0:
            return None
        s = Stack()
        i = 0
        cur = pRoot
        while s.isEmpty() == False or cur != None:
            while cur != None:
                s.push(cur)
                cur = cur.left
            cur = s.pop()
            i += 1
            if k == i:
                return cur
            cur = cur.right
        return None
```



# 滑动窗口的最大值

输入一个list和一个size，以size为长的窗口在list上滑动，输出每一次滑动时窗口内的最大值。 如输入数组[2,3,4,2,6,2,5,1],size为3.那么一共有6个窗口，他们的最大值分别为[4,4,6,6,6,5]。

```python
# 不用任何数学函数的版本，利用q和p作为窗口的第一个和最后一个值，用max_val储存上个窗口的最大值，用index代表上个窗口最大值的索引，每次滑动时，如果新的值大于max_val，则新的值最大。如果新的值不大于max_val，则需判断max_val是否还在窗口内，如果在，直接到下一个窗口，如果不在，则重新计算当前范围内最大值。
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if num is None or size == 0:
            return []
        if size > len(num):
            return []
        result = []
        q = 0
        p = size - 1
        start = 1
        max_v = 0
        index = 0
        while p < len(num):
            if max_v <= num[p] and start == 0:
                max_v = num[p]
                index = p
                result.append(max_v)
                p += 1
                q += 1
            else:
                if q <= index <= p and start == 0:
                    result.append(max_v)
                    q += 1
                    p += 1
                    continue
                else:
                    max_v = num[q]
                    index = q
                    for i in range(q+1,p+1):
                        if max_v < num[i]:
                            max_v = num[i]
                            index = i
                    result.append(max_v)
                    p += 1
                    q += 1
                    start = 0
        return result
```

