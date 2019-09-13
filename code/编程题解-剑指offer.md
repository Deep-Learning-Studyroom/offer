 # 编程题解-剑指offer

[TOC]

# 数组中的逆序对



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

# 数组中重复的数字

1. 排序后遍历找重复数字 O(nlogn)
2. 用一个字典来找重复数字 时间和空间都是O(n)
3. 时间O(n), 空间O(1)。 方法是遍历list，使每一个空间都保存与下标同样的值。详情见剑指offer书

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        n = len(numbers)
        if n == 0:
            return False
        for i in range(n):
            if numbers[i] > n - 1:
                return False
            while numbers[i] != i:
                if numbers[numbers[i]] == numbers[i]:
                    duplication[0] = numbers[i]
                    return True
                else:
                    t = numbers[numbers[i]]
                    numbers[numbers[i]] = numbers[i]
                    numbers[i] = t
        return False
```

# 在二维数组中查找数字

​	输入给一个二维矩阵，这个矩阵的数字排序是从左向右递增、从上往下递增的。给定一个目标数字，判断这个数字是否存在矩阵内。做法是取矩阵的左下角或右上角，必须取这两个点，因为这两个点所在的行和列分别是大于或小于它们的，下面的代码选择的左下角。当左下角的数字大于目标数字时，则将左下角数字的坐标行-1，若小于目标数字时，则把左下角数字的列+1。

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if len(array) == 0:
            return False
        row = len(array)
        col = len(array[0])
        left_down_col = 0
        left_down_row = row - 1
        while left_down_row >= 0 and left_down_col < col:
            if array[left_down_row][left_down_col] == target:
                return True
            elif array[left_down_row][left_down_col] < target:
                left_down_col += 1
            else:
                left_down_row -= 1
        return False
```

# 替换空格

将输入的字符串中所有的空格替换为”%20“，这道题用Python实现起来比较容易，但是和C++原生数组的话差的也不多，如果是C++的话需要提前分配好空间才可以插入。思路是对字符串从尾部开始，将这些值用头插法赋值给result，如果碰到空格，则连续插入%20。直接用list的insert方法会比较方便。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        result = list()
        i = len(s) - 1
        while i >= 0:
            if s[i] != ' ':
                result.insert(0, s[i])
                i -= 1
            else:
                result.insert(0, "0")
                result.insert(0, "2")
                result.insert(0, "%")
                i -= 1
        result = "".join(result)
        return result
```

# 重建二叉树

给定一个二叉树的前序遍历和中序遍历的list，把二叉树重建出来。思路就是递归，首先前序遍历的第一个数字是根节点，然后在中序遍历中找到这个数字，这个数字的左边是左子树，右边是右子树，这样在中序遍历中获得左右子树的长度。在前序遍历中用这个长度就可以获得左右子树的前序遍历，现在分别获得了左右子树的前序遍历和中序遍历，就可以分别对左右子树再次递归这一方法了。

```python
class TreeNode:
	def __init__(self, x):
      self.val = x
      self.left = None
      self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre or not tin:
            return -1
        length = len(pre)
        return self.ConstructCore(pre, tin)
    
    def ConstructCore(self, pre, tin):
        root = TreeNode(pre[0])
        if len(pre) == 1:
            return root
        for i in range(len(tin)):
            if tin[i] == pre[0]:
                break
        if i > 0:
            root.left = self.ConstructCore(pre[1:i+1], tin[0:i])
        if i < len(tin) - 1:
            root.right = self.ConstructCore(pre[i+1:], tin[i+1:])
        return root
```

# 二叉树的下一个节点

给定一个树的某一个节点，判断在中序遍历下，下一个节点是什么，这棵树除了左右子树指针，还有一个指向父节点的指针。要明确有三种情况，

1. 当这个节点有右子树时，下一个节点是它的右子树的最左边节点，用一个while即可。
2. 当这个节点没有右子树时，那就只有往上一直遍历父节点，只有当某一个节点是其父节点的左子树时，下一个节点就是父节点。
3. 如果其上每一个父节点都是更高层父节点的右子树，一直延续到根节点，那么下一个节点为空。

```python
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return None
        if pNode.right:
            p = pNode.right
            while p.left:
                p = p.left
            return p
        else:
            cur = pNode
            parent = pNode.next
            while parent and parent.right == cur:
                cur = parent
                parent = parent.next
            return parent
```

# 旋转数组的最小数字

给定一个递增数组的旋转数组，找到数组中最小的数字。因为旋转数组一般来说有两个有序的子数组，并且最小的数字刚好是这两个子数组的分界。所以使用二分查找的方式来寻找这个数字。注意这个代码也不是完美的，当p1、p2、mid三个指针指向的数字一样大时，如[0, 1, 1, 1, ,1]的旋转，那这种方法就会失效，只能改用顺序查找的方式。

```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        length = len(rotateArray)
        if length == 0:
            return 0
        p1 = 0
        p2 = length - 1
        if rotateArray[p1] < rotateArray[p2]:
            return rotateArray[p1]  # 如果第一个数小于最后一个数，说明数组是有序的，直接返回第一个数。
        # 当两个指针相邻时，最小的数字是右边那个指针
        while p1 + 1 != p2:
            mid = int((p1 + p2) / 2)
            if rotateArray[p1] <= rotateArray[mid]:
                p1 = mid
            else:
                p2 = mid
        return rotateArray[p2]
```

# 矩阵中的路径

回溯法

```python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        cur_path = 0
        path = list(path)
        matrix = list(matrix)
        flag = []
        for _ in range(rows * cols):
            flag.append(0)
        for row in range(rows):
            for col in range(cols):
                if self.hasPathCore(matrix, rows, cols, row, col, flag, path, cur_path):
                    return True
        return False

    def hasPathCore(self, matrix, rows, cols, row, col, flag, path, cur_path):
        index = row * cols + col
        if row < 0 or col < 0 or row >= rows or col >= cols or \
                matrix[index] != path[cur_path] or flag[index] == 1:
            return False
        if cur_path == len(path) - 1:
            return True
        flag[index] = 1
        if self.hasPathCore(matrix, rows, cols, row - 1, col, flag, path, cur_path + 1) or \
           self.hasPathCore(matrix, rows, cols, row + 1, col, flag, path, cur_path + 1) or \
           self.hasPathCore(matrix, rows, cols, row, col - 1, flag, path, cur_path + 1) or \
           self.hasPathCore(matrix, rows, cols, row, col + 1, flag, path, cur_path + 1):
            return True
        flag[index] = 0
        return False


if __name__ == '__main__':
    matrix = "ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS"
    rows = 5
    cols = 8
    path = "SLHECCEIDEJFGGFIE"
    print(Solution().hasPath(matrix, rows, cols, path))
```

# 机器人的运动范围

仍然是回溯法， 有一个地方要注意，python中将list作为函数参数时，是形式引用，所以代码中的flag在全局是共享的。

```python
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        if rows == 0 and cols == 0:
            return 0
        flag = list()
        for i in range(cols * rows):
            flag.append(0)

        count = self.movingCountCore(threshold, rows, cols, 0, 0, flag)
        return count

    def movingCountCore(self, threshold, rows, cols, row, col, flag):
        index = row * cols + col
        if row < 0 or col < 0 or row >= rows or col >= cols or \
                flag[index] == 1 or not self.check(threshold, row, col):
            return 0
        flag[index] = 1
		    
        # flag是共享的
        return 1 + self.movingCountCore(threshold, rows, cols, row - 1, col, flag) + \
               self.movingCountCore(threshold, rows, cols, row + 1, col, flag) + \
               self.movingCountCore(threshold, rows, cols, row, col - 1, flag) + \
               self.movingCountCore(threshold, rows, cols, row, col + 1, flag)

    def check(self, threshold, row, col):
        """ 检查当前点是否合规"""
        sum_ = 0
        while row % 10 != row:
            sum_ += row % 10
            row = int(row / 10)
        sum_ += row
        while col % 10 != col:
            sum_ += col % 10
            col = int(col / 10)
        sum_ += col
        if sum_ > threshold:
            return False
        else:
            return True

if __name__ == '__main__':
    print(Solution().movingCount(5, 10, 10))
```

# 删除链表中的重复结点

这里删除重复结点不是说把重复部分删掉就行了，而是只要重复就全删了。例如【1，2，2，3】删除后是【1，3】而不是【1，2，3】，做法是要另外设一个Head结点，先让它指向真实头结点。然后设pre=Head， pre的含义是当前不重复部分的最后一个结点，然后设一个工作指针p，当p和p.next值相等时，则要一直向后遍历，找到下一个不重复的数字，然后让pre指向它。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if not pHead:
            return None
        if pHead.next is None:
            return pHead
        Head = ListNode(0)
        Head.next = pHead
        pre = Head
        p = pHead
        while p:
            if p.next and p.val == p.next.val:
                while p.next and p.val == p.next.val:
                    p = p.next
                pre.next = p.next
                p = p.next
            else:
                pre = p
                p = p.next
        return Head.next
```



# 正则表达式匹配

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        if not s or not pattern:
            return False
        si = 0
        pi = 0
        return self.matchCore(s, pattern, si, pi)

    def matchCore(self, s, pattern, si, pi):
        if si == len(s) and pi == len(pattern):
            return True
        if si < len(s) and pi == len(pattern):
            return False
        if pi < len(pattern) - 1 and pattern[pi + 1] == "*":
            if (si < len(s) and s[si] == pattern[pi]) or (si < len(s) and pattern[pi] == "."):
                return self.matchCore(s, pattern, si + 1, pi) or \
                       self.matchCore(s, pattern, si + 1, pi + 2) or \
                       self.matchCore(s, pattern, si, pi + 2)
            else:
                return self.matchCore(s, pattern, si, pi + 2)
        elif (si < len(s) and s[si] == pattern[pi]) or (pattern[pi] == "." and si < len(s)):
            return self.matchCore(s, pattern, si + 1, pi + 1)
        else:
            return False
```

动态规划：

```python
class Solution:
    def isMatch(self, s: str, pattern: str) -> bool:
        len_s = len(s)
        len_p = len(pattern)
        dp = []
        for i in range(len_s+1):
            dp.append([0] * (len_p+1))
        dp[0][0] = 1
        for i in range(1, len_p+1):
            dp[0][i] = i > 1 and "*" == pattern[i-1] and dp[0][i-2]

        for i in range(1, len_s+1):
            for j in range(1, len_p+1):
                if s[i-1] == pattern[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                elif pattern[j-1] == ".":
                    dp[i][j] = dp[i-1][j-1]
                elif pattern[j-1] == "*":
                    if pattern[j-2] == s[i-1] or pattern[j-2] == ".":
                        dp[i][j] = dp[i][j-2] or dp[i][j-1] or dp[i-1][j]
                    else:
                        dp[i][j] = dp[i][j-2]
        return dp[-1][-1]
```



# 表示数值的字符串

此题的关键在于始终用一个numeric来标识：到字符串的目前位置为止，前面的字符是否符合数值规范。然后数值可以被E或e隔开，E的前后都可以是一个完整数值。完整数值的第一为可以是+或者-号， 也可以是数字，当向后遍历出现不是数值的字符时，这个字符必须是”.”、E、e、或者是空（即字符串遍历结束），若已经遇见过E、e或者小数点，则后面只能出现数字（第一位可以是正负号）。 判断数值使使用ord（）函数判断Ascii码，在遍历的过程中不断对s进行切片操作，以模拟指针，也可以用一个index来模拟指针。另外注意在对s进行切片前，要判断长度，防止越界，如果当前位是最后一位，则将s置空。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        if s is None:
            return False
        numeric, s = self.scanInteger(s)
        if s is not None and s[0] == ".":
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
            temp, s = self.scanUnsignedInteger(s)
            numeric = temp or numeric

        if s is not None and (s[0] == "E" or s[0] == "e"):
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
            temp, s = self.scanInteger(s)
            numeric = temp and numeric
        return numeric and s is None

    def scanInteger(self, s):
      """判断完整数字，先判断符号，再判断没有符号的数字"""
        if s is not None and (s[0] == "+" or s[0] == "-"):
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
        return self.scanUnsignedInteger(s)

    def scanUnsignedInteger(self, s):
      """判断没有符号的数字，只要出现了一次数字，flag即为True，一直遍历到第一次遇见非数值为止"""
        flag = False  # 是否有若干数字的标识
        while s is not None and ord(s[0]) >= 48 and ord(s[0]) <= 57:
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
            flag = True
        return flag, s
```

# 调整数组顺序使奇数位于偶数前面

牛客上的题目和书上的题目不完全一样，牛客上题目是：给定一个list，将所有奇数调到偶数的前面，并且奇数之间、偶数之间数字的排列顺序不变。解题方案，直接使用python list的insert和del方法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        if len(array) == 0:
            return []
        cur = 0
        for i in range(0, len(array)):
            if array[i] % 2 == 1:
                array.insert(cur, array[i])
                del array[i+1]
                cur += 1
        return array
```

# 链表中倒数第k个节点

设两个指针p1、p2， p1先向前移动k-1步，然后同时移动p1、p2，直到p1移动到最后一位，这样p2所在节点就是链表倒数第k个节点。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if head is None:
            return None
        if k <= 0:
            return None
        p1 = head
        p2 = head
        for _ in range(k-1):
            if p1.next is None:
                return None
            p1 = p1.next
        while p1.next is not None:
            p1 = p1.next
            p2 = p2.next
        return p2
```



