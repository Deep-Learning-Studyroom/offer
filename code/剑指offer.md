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

解法3：这道题和上一道题的区别在于**一定范围内数字的个数不够**。所以可以利用类似二分查找的方法做。时间复杂度为O(nlog(n))，空间复杂度为O(1)。

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

# 面试题4：二维数组中的查找

>在一个二维数组中，每一行都按照从左到右递增的顺序，每一列都按照从上到下递增的顺序。请完成这样一个函数，输入这样一个二维数组和一个整数，判断数组中是否含有该整数。

解法：讲num和arr的右上角的数字比较，如果num更小，那么最右边那一列就不考虑了；如果num更大，那么就在最右边那一列寻找。比较num和右下角的数字，如果num更大，那么数组中不存在num这个数字；如果num小于等于右下角的数字，那么就遍历最右边那一列的数字，找不到就不存在，否则就存在。

```python
import numpy as np
def find(arr, num):
    """
    arr是numpy的数组
    """
    m, n = arr.shape
    print(m, n)
    
    while(num < arr[0, n-1]):
        n -= 1
        print(n)
    if num > arr[m-1, n-1]:
        print("num {} is not in arr".format(num))
        return 0
    else:
        for value in arr[0:-1, n-1]:
            if value == num:
                print("num {} is in arr".format(num))
                return 1
        print("num {} is not in arr".format(num))
        return 0

arr = np.array([[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]])
find(arr, 8)
```

# 面试题5：替换空格
>请实现一个函数，把字符串中的每个空格替换成"%20"。例如，输入"We are happy."，则输出"We%20are%20happy."。

思考：原来1个字符，替换之后变为3个，字符串会边长。如果是在原来的字符串上替换，那么会覆盖掉空格后面的内存；如果创建新的字符串进行替换，那么我们需要足够的内存。这一点可以和面试官交流，问清楚内存是否足够。假设面试官让我们在原来的字符串上进行修改，并保证输入字符串后面有足够的内存。

解法1：从头到尾扫描字符串，每次碰到空格进行替换。时间复杂度O(n2)，不足以拿到offer。

解法2：先遍历一遍字符串，找到空格个数，然后得到最终字符串的长度（原字符串长度+两倍的空格数）。然后用两个指针p1, p2分别指向原字符串的末尾和替换后字符串的末尾。然后p1从后往前遍历，遇到不是空格的字符就复制到p2对应的位置，然后p1, p2同事往前走一步（减1）；如果p1遇到空格，那么就把%20插入到p2之前，然后p2向前移动3格，p1移动1格。**关键在于：从后往前**。时间复杂度O(n)。

```python
def replace_blank(string):
    """
    python中字符串不能更改，因此这里转换为数组进行处理
    """
    string = list(map(str, string))
    if string is None:
        return
    
    original_length = len(string)
    blank_number =0
    for i in string:
        if i == " ":
            blank_number += 1
    new_length = original_length + 2 * blank_number

    for i in range(2 * blank_number):
        string.append(0)

    p1 = original_length - 1
    p2 = new_length - 1

    while p1 != p2:
        if string[p1] != " ":
            string[p2] = string[p1]
            p1 -= 1
            p2 -= 1
        else:
            string[p2] = "0"
            string[p2 - 1] = "2"
            string[p2 - 2] = "%"
            p1 -= 1
            p2 -= 3
    return "".join(string)

print(replace_blank("Good moning"))
```

## 相关题目
>有两个排序的数组A1和A2，内存在A1的末尾有足够的空余空间容纳A2。请实现这样一个函数，把A2中的所有数字插入到A1中，并且所有的数字是排序的。

思考：假设这里的排序是从小到大。和前面的题一样，直观的思维是从前往后比较，然后插入，但是这样就会出现一个元素复制多次的情况。更好的方法是从尾到头比较两个数组中的数字，然后把较大的数字复制到A1中的合适位置。

```python
def test(A1, A2):
    """
    A1 and A2 both are sorted arrays, 
    combine A2 into A1 and the new array 
    should be sorted.
    """

    length_1 = len(A1)
    length_2 = len(A2)
    new_length = length_1 + length_2
    for i in range(length_2):
        A1.append(' ')
    
    p = new_length - 1
    p1 = length_1 - 1
    p2 = length_2 - 1

    while p1 >= 0 and p2 >= 0:
        if A1[p1] > A2[p2]:
            A1[p] = A1[p1]
            p -= 1
            p1 -= 1
            print(A1)
        else:
            A1[p] = A2[p2]
            p -= 1
            p2 -= 1
            print(A1)
    return A1

print(test([1,2,3,4,5], [2,4,6]))
```
# 面试题6：从尾到头打印链表

三种方法：栈、列表逆序和递归。基于递归的方法有一个不足：当链表非常长时，就会导致函数调用的层级很深，从而有可能导致函数调用栈溢出。不如前两种的鲁棒性更好。

```python 
#class ListNode:
#    def __init__(self, x):
#        self.val = x
#        self.next = None

class Solution:
    """
    使用列表逆序的方法
    """
    def print_list_form_tail_to_head(self, list_node):
        out = []
        if list_node is None:
            return out
        while list_node.next is not None:
            out.append(list_node.val)
            list_node = list_node.next
        out.append(list_node.val)
        out.reverse()
        return out

class Solution:
    """
    使用栈的方法
    """
    def print_list_form_tail_to_head(self, list_node):
        out = []
        if list_node is None:
            return out
        while list_node.next is not None:
            out.append(list_node.val)
            list_node = list_node.next
        out.append(list_node.val)

        result = []
        while len(out) != 0:
            result.append(out.pop())
        return out

class Solution:
    """
    使用递归的方法
    """
    def print_list_form_tail_to_head(self, list_node):
        out = []
        if list_node is None:
            return out

        return self.print_list_form_tail_to_head(list_node.next) + [list_node.val]

```

# 面试题7：重建二叉树

## 二叉树重要基础知识

由于树的操作设计大量指针，因此关于树的面试题大都不容易，也是面试官喜欢考察的点。

**二叉树最重要的操作是遍历，分为先序遍历（根左右）、中序遍历（左根右）和后序遍历（左右根）。每个操作都有递归和堆栈两种实现方法。另外，还有宽度优先遍历（又称为层次遍历），即按照从上到下、每一层内从左到右的顺序遍历。这7个方法必须熟练掌握！**

**遍历**：  
- 深度优先遍历:  
    - 先序遍历：**递归**或堆栈  
    - 中序遍历：**递归**或堆栈  
    - 后序遍历：**递归**或堆栈  
- 广度优先遍历:  
    - 层次遍历：**队列**  
    
二叉树的特例：  
    - 二叉搜索树：查找的时间复杂度O(logn)。参考面试题36“二叉搜索树与双向链表”和面试题68“树中两个节点的最低公共祖先”。  
    - 堆：最大堆、最小堆。参考面试题40“最小的k个数”。  
    - 红黑树：树的节点有红黑两种颜色，并且从根节点到叶节点的最长路径不超过最短路径的两倍。参考面试题40“最小的k个数”。  


 ```python
class Node(object):
    """
    节点类
    """
    def __init__(self, elem=-1, lchild=None, rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild

class Tree(object):
    """
    树类
    """
    def __init__(self):
        self.root = Node()
        self.myQueue = []

    def add(self, elem):
        """
        为树添加节点
        """
        node = Node(elem)

        if self.root.elem == -1:            # 如果树是空的，则对根节点幅值
            self.root = node
            self.myQueue.append(self.root)

        else:
            treeNode = self.myQueue[0]      # 此节点的子树还没有齐
            if treeNode.lchild == None:
                treeNode.lchild = node
                self.myQueue.append(treeNode.lchild)
            else:
                treeNode.rchild = node
                self.myQueue.append(treeNode.rchild)
                self.myQueue.pop(0)         # 如果该节点存在左右子树，将此节点丢弃

    def front_recursive(self,root):
        """
        利用递归实现前序遍历（根左右）
        """
        if root is None:
            return
        print(root.elem)
        self.front_recursive(root.lchild)
        self.front_recursive(root.rchild)

    def middle_recursive(self, root):
        """
        利用递归实现中序遍历(左根右)
        """
        if root is None:
            return
        self.middle_recursive(root.lchild)
        print(root.elem)
        self.middle_recursive(root.rchild)

    def later_recursive(self, root):
        """"
        利用递归实现后序遍历（左右根）
        """
        if root is None:
            return
        self.later_recursive(root.lchild)
        self.later_recursive(root.rchild)
        print(root.elem)

    def front_stack(self, root):
        """利用堆栈实现树的先序遍历"""
        if root is None:
            return
        my_stack = []
        node = root
        while node or my_stack:
            while node:                  # 从根节点开始，一直找它的左子树
                print(node.elem)
                my_stack.append(node)
                node = node.lchild
            node = my_stack.pop()        # while结束表示当前节点node为空，即前一个节点没有左子树了
            node = node.rchild           # 开始查看它的右子树

    def middle_stack(self, root):
        """利用堆栈实现树的中序遍历"""
        if root is None:
            return
        my_stack = []
        node = root
        while node or my_stack:
            while node:                         # 从根节点开始，一直找它的左子树
                my_stack.append(node)
                node = node.lchild
            node = my_stack.pop()               #  while结束表示当前节点node为空，即前一个节点没有左子树了
            print(node.elem)
            node = node.rchild                  # 开始查看它的右子树

    def later_stack(self, root):
        """利用堆栈实现树的后序遍历"""
        if root is None:
            return

        my_stack1 = []
        my_stack2 = []
        node = root
        my_stack1.append(node)
        while my_stack1:                         # 这个while循环的功能是找出后序遍历的逆序，存在my_stack2里面
            node = my_stack1.pop()
            if node.lchild:
                my_stack1.append(node.lchild)
            if node.rchild:
                my_stack1.append(node.rchild)
            my_stack2.append(node)
        while my_stack2:                         # 将my_stack2中的元素出栈，即为后序遍历的顺序
            print(my_stack2.pop().elem)

    def level_queue(self, root):
        """利用队列实现树的层次遍历"""
        if root is None:
            return
        my_queue = []
        node = root
        my_queue.append(node)

        while my_queue:
            node = my_queue.pop(0)
            print(node.elem)
            if node.lchild != None:
                my_queue.append(node.lchild)
            if node.rchild != None:
                my_queue.append(node.rchild)

if __name__ == '__main__':
    """主函数"""
    elems = range(10)           #生成十个数据作为树节点
    tree = Tree()          #新建一个树对象
    for elem in elems:
        tree.add(elem)           #逐个添加树的节点

    print('队列实现层次遍历:')
    tree.level_queue(tree.root)

    print('\n\n递归实现先序遍历:')
    tree.front_recursive(tree.root)
    print('\n\n堆栈实现先序遍历:')
    tree.front_stack(tree.root)

    print('\n递归实现中序遍历:')
    tree.middle_recursive(tree.root)
    print('\n堆栈实现中序遍历:')
    tree.middle_stack(tree.root)

    print('\n递归实现后序遍历:')
    tree.later_recursive(tree.root)
    print('\n堆栈实现后序遍历:')
    tree.later_stack(tree.root)
```   

## 重建二叉树题目和解法

>输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如，输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建如下图所示的二叉树并输出它的头节点。

![](https://github.com/Deep-Learning-Studyroom/offer/blob/master/pictures/reconstruct_binary_tree.PNG) 

解法：如图所示，前序遍历的第一个数字1是根节点的值，然后去中序遍历序列中找到1的位置，1之前的是左子树的中序遍历序列（节点数为3），之后的是右子树的中序遍历序列（节点数为4）。然后根据左子树的节点数和右子树的节点数可以在前序遍历序列中找到左子树的前序遍历序列和右子树的前序遍历序列。**这样在两个序列中找到了左右子树对应的前序遍历序列和中序遍历序列。**
接下来可以用**递归**的方法做。

```python

# -*- coding:utf-8 -*-
'''
重建二叉树
题目描述
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        if len(pre) == 1:
            return TreeNode(pre[0])
        else:
            # 每棵子树的根节点肯定是pre子数组的首元素，所以每次新建一个子树的根节点
            res = TreeNode(pre[0])
            res.left = self.reConstructBinaryTree(pre[1: tin.index(pre[0]) + 1], tin[: tin.index(pre[0])])
            res.right = self.reConstructBinaryTree(pre[tin.index(pre[0]) + 1: ], tin[tin.index(pre[0]) + 1: ])
        return res
```

# 面试题8：二叉树的下一个节点

>给定一颗二叉树和其中的一个节点，如何找出中序遍历序列的下一个节点？树中的节点除了有两个分别指向左、右子节点的指针，还有一个指向父节点的指针。

解法：  
- 如果一个节点有右子树，那么它的下一个节点就是**其右子树的最左节点**。  
- 如果一个节点没有右子树，并且它是其父节点的左子节点，那么它的下一个节点就是**其父节点**。    
- 如果一个节点没有右子树，并且它是其父节点的右子节点，那么**沿着指向父节点的指针一直向上遍历，直到找到一个是它父节点的左子节点的节点。如果这个节点存在，那么这个节点的父节点就是我们要找的下一个节点；如果这个节点不存在，那么该节点没有下一个节点**

```python
class BinaryTreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
class Solution:
    def __init__(self, binary_tree, node):
        self.binary_tree = binary_tree
        self.node = node

    def find_next(self):
        if self.binary_tree is None:
            return None

        # 存在右子树
        if self.node.right is not None:
            next_node = self.node.right
            while next_node.left is not None:
                next_node = next_node.left
            return next_node

        else:
            # 沿着指向父节点的指针一直向上遍历，直到找到一个是它父节点的左子节点的节点。
            # 如果这个节点存在，那么这个节点的父节点就是我们要找的下一个节点，否则返回None
            # 如果是父节点的左子节点
            if self.node.parent.left == self.node:
                return self.node.parent
            # 如果是父节点的右子节点
            else:
                next_node = self.node.parent
                while next_node is not None:
                    if next_node.parent.left == next_node:
                        return next_node.parent
                    next_node = next_node.parent
                return None
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