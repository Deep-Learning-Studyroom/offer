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

