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

# 面试题9：用两个栈实现队列

>用两个栈实现队列。实现队列的两个功能，push（队列尾部插入节点）和pop（队列头部删除节点）。

解法：两个栈，stack1和stack2.每次push的时候都从stack1 append即可，每次pop时如果stack2有元素，就从stack2 pop；如果stack2没有元素，那么就把stack1的所有元素逆序append进stack2，然后stack2 pop即可。


```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
```


# 面试题10：斐波那契数列(Fibonacci) 

## 斐波那契数列

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

## 补充题：青蛙跳台阶

>一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个n级台阶总共有多少总做法。

解法：第n级台阶的方法数 = 第n-1级台阶的方法数 + 第n-2级台阶的方法数，所以就是斐波那契数列

>拓展：把青蛙跳的方式改为，既可以跳上1级台阶，也可以跳上2级台阶，也可以跳上n级台阶，那么跳上n级台阶总共有多少总方法？用数学归纳法可以得到，f(n) = 2^(n-1)


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

# 面试题11：旋转数组的最小数字

>把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的旋转，输出旋转数组的最小元素。例如，数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，最小的元素为1.

解法：二分查找。定义start和end是数组第一个和最后一个的索引，然后middle是两者和除以2的商。如果middle对应的数字大于等于start对应的数字，那么说明middle在前半部分，则把middle的值赋给start；否则说明middle在后半段，middle值赋给end。循环直到end-start==1。此时end对应的数就是答案。


```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if not rotateArray:
            return 0
        
        start = 0
        end = len(rotateArray) - 1
        
        while (end - start) > 1:
            middle = (start + end) // 2
            if rotateArray[middle] >= rotateArray[start]:
                start = middle
            else:
                end = middle
        return rotateArray[end]
```

# 面试题12：矩阵中的路径

>请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左右上下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。

解法：**通常在二维矩阵中找路径都可以用回溯法解决**。

```python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        matrix = list(matrix)
        path = list(path)
        cur_path = 0
        flags = [0] * len(matrix)

        for row in range(rows):
            for col in range(cols):
                if self.find(matrix, rows, cols, row, col, path, cur_path, flags):
                    return True
        return False

    def find(self, matrix, rows, cols, row, col, path, cur_path, flags):
        index = row * cols + col

        if col < 0 or col > cols -1 or row < 0 or row > rows - 1 or flags[index]  == 1 or matrix[index] != path[cur_path]:
            return False
        if cur_path == len(path) - 1:
            return True

        flags[index] = 1
        if self.find(matrix, rows, cols, row + 1, col, path, cur_path + 1,flags) or \
            self.find(matrix, rows, cols, row - 1, col, path, cur_path + 1, flags) or \
            self.find(matrix, rows, cols, row, col + 1, path, cur_path + 1, flags) or \
            self.find(matrix, rows, cols, row, col - 1, path, cur_path + 1, flags):
            return True
        flags[index] = 0
        return False

solution = Solution()

print(solution.hasPath("ABCESFCSADEE",3,4,"SEC"))
print(solution.hasPath("ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS",5,8,"SGGFIECVAASABCEHJIGQEM"))
print(solution.hasPath("ABTGCFCSJDEH",3,4,"BFCE"))
print(solution.hasPath("ABTGCFCSJDEH",3,4,"BFCJ"))
```




# 面试题32：从上到下打印二叉树

**题目一：不分行从上到下打印二叉树**

>从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

解法：二叉树的层次遍历，利用队列

**题目二：分行从上到下打印二叉树**

解法：添加两个值，分别用来记录当前层待打印的节点数量和下一层的节点数量

**题目三：之字形打印二叉树**

解法：在分行打印的基础上修改，添加一个记录的列表all和一个层数level_num，列表中的第n个元素是第n层(这里层数也从0开始)节点从左到右组成的列表。分行打印算法中打印的地方改为想列表all的对应元素中append元素。打印的时候偶数层的列表逆序一下再打印即可。同时要注意打印换行。

代码：

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

    def print_binary_tree(self):
        """不分行从上到下打印二叉树"""
        if self.root is None:
            return None

        my_queue = []
        node = self.root
        my_queue.append(node)

        while my_queue:
            node = my_queue.pop(0)
            print(node.elem, end=" ")
            if node.lchild is not None:
                my_queue.append(node.lchild)
            if node.rchild is not None:
                my_queue.append(node.rchild)

    def print_binary_tree2(self):
        """分行从上到下打印二叉树，需要额外另个量来记录本层待打印的节点数量以及下一层的节点数量"""
        if self.root is None:
            return None

        my_queue = []
        node = self.root
        my_queue.append(node)
        to_be_printed = 1
        next_level_num = 0

        while my_queue:
            node = my_queue.pop(0)
            if to_be_printed > 0:
                print(node.elem, end=" ")
                to_be_printed -= 1
            else:
                print()
                to_be_printed = next_level_num
                next_level_num = 0
                print(node.elem, end=" ")
                to_be_printed -= 1

            if node.lchild is not None:
                my_queue.append(node.lchild)
                next_level_num += 1
            if node.rchild is not None:
                my_queue.append(node.rchild)
                next_level_num += 1

    def print_binary_tree3(self):
        """之字形打印二叉树，在分行打印的基础上添加一个记录本层顺序的值，之前打印的地方换成压到一个列表"""
        if self.root is None:
            return None

        my_queue = []
        node = self.root
        my_queue.append(node)
        to_be_printed = 1
        next_level_num = 0
        all = [[],]
        level_num = 0

        while my_queue:
            node = my_queue.pop(0)
            if to_be_printed > 0:
                #print(node.elem, end=" ")
                all[level_num].append(node.elem)
                to_be_printed -= 1
            else:
                print()
                to_be_printed = next_level_num
                next_level_num = 0
                level_num += 1
                all.append([])
                #print(node.elem, end=" ")
                all[level_num].append(node.elem)
                to_be_printed -= 1

            if node.lchild is not None:
                my_queue.append(node.lchild)
                next_level_num += 1
            if node.rchild is not None:
                my_queue.append(node.rchild)
                next_level_num += 1

        for i in range(len(all)):
            if i % 2 == 0:
                for j in all[i]:
                    print(j, end=" ")
                if i != len(all) - 1:
                    print()
            else:
                all[i].reverse()
                for j in all[i]:
                    print(j, end=" ")
                if i != len(all) - 1:
                    print()



if __name__ == '__main__':
    """主函数"""
    elems = range(10)              #生成十个数据作为树节点
    tree = Tree()                  #新建一个树对象
    for elem in elems:
        tree.add(elem)             #逐个添加树的节点

    print('不分行从上到下打印二叉树')
    tree.print_binary_tree()

    print('分行从上到下打印二叉树')
    tree.print_binary_tree2()

    print('之字形打印二叉树')
    tree.print_binary_tree3()
```


# 面试题33：二叉搜索树的后序遍历序列

>输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回true，否则返回false。假设输入的数组的任意两个数字都互不相同。

解法：

```python
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False
        length = len(sequence)
        # BST后序遍历的最后一个值是root
        root = sequence[-1]
        indexRight = 0
        for i in range(0, length - 1):
            if sequence[i] > root:
                break
            indexRight += 1
        for j in range(indexRight + 1, length - 1):
            if sequence[j] < root:
                return False
        # 递归检查左子树是否可以为BST
        left = True
        if indexRight > 1:
            left = self.VerifySquenceOfBST(sequence[:indexRight])
        
        # 递归检查右子树是否可以为BST
        right = True
        if indexRight < length - 1:
            right = self.VerifySquenceOfBST(sequence[indexRight:length - 1])
         
        return left and right
```

# 面试题34：二叉树中和为某一值的路径

>输入一个二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
        
        # 只有根节点且根节点等于expectNumber的情况
        if root and not root.left and not root.right and root.val == expectNumber:
            return [[root.val]]
        
        # 递归
        res = []
        left = self.FindPath(root.left, expectNumber - root.val)
        right = self.FindPath(root.right, expectNumber - root.val)
        
        for i in left + right:
            res.append([root.val] + i)
        
        return res
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