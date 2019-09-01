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

解法：首先比较target和array最小和最大的数，看看是不是在两者之间，否则就返回False。然后while循环，比较target是否小于右上角的数，
如果小于，就把cols减一。然后比较最右边的一列和target的大小，如果有相等的就返回True，如果有大于target的就遍历当前行，如果有相等的就返回
True。如果遍历结束还没有返回，就返回False。

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array)
        cols = len(array[0])
        if not rows or not cols or target > array[rows - 1][cols - 1] or target < array[0][0]:
            return False
        while target < array[0][cols - 1]:
            cols -= 1
        for i in range(rows):
            if array[i][cols - 1] == target:
                return True
            elif array[i][cols - 1] > target:
                for j in range(cols):
                    if array[i][j] == target:
                        return True
        return False
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

# 面试题13：机器人的运动范围

>地上有一个m行n列的方格。一个机器人从坐标(0,0)的格子开始移动，它每次可以向左、右、上、下移动一格，但不能进入行坐标和列坐标的位数之和大于k的格子。请问该机器人能够到达多少个格子？

解法:：回溯法。仿照“矩阵中的路径”的解法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        if threshold == 0:
            return 1
        res = []
        flags = [0] * (rows * cols)
        col = 0
        row = 0
        return len(self.find(threshold, rows, cols, row, col, flags, res))

    def find(self, threshold, rows, cols, row, col, flags, res):
        #print('*')
        index = row * cols + col
        if col < 0 or col > cols - 1 or row < 0 or row > rows - 1 or flags[index] == 1 or self.mySum(col) + self.mySum(row) > threshold:
            return res
        flags[index] = 1
        res.append((row, col))
        #print((row, col))

        res = self.find(threshold, rows, cols, row - 1, col, flags, res)
        res = self.find(threshold, rows, cols, row + 1, col, flags, res)
        res = self.find(threshold, rows, cols, row, col - 1, flags, res)
        res = self.find(threshold, rows, cols, row, col + 1, flags, res)

        return res

    def mySum(self, int1):
        sum1 = 0
        while int1 > 0:
            sum1 += int1 % 10
            int1 = int1 // 10
        return sum1

solution = Solution()
print(solution.movingCount(5,10,10))
print(solution.movingCount(15,20,20))
```
优化后的代码如下(去掉flags数组，精简代码)：

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.res = set()
    def movingCount(self, threshold, rows, cols):
        # write code here
        return self.find(threshold, rows, cols, 0, 0)

    def find(self, threshold, rows, cols, row, col):
        index = row * cols + col
        if col < 0 or col > cols - 1 or row < 0 or row > rows - 1 or (row, col) in self.res or self.mySum(col) + self.mySum(row) > threshold:
            return 0
        self.res.add((row, col))
        return 1 + self.find(threshold, rows, cols, row, col-1) + self.find(threshold, rows, cols, row, col+1) + self.find(threshold, rows, cols, row-1, col) + self.find(threshold, rows, cols, row+1, col)

    def mySum(self, int1):
        sum1 = 0
        while int1 > 0:
            sum1 += int1 % 10
            int1 = int1 // 10
        return sum1
```

# 面试题14：剪绳子

>给你一根长度为n的绳子，请把绳子剪成m段（m, n都是整数，n > 1 并且 m > 1），每段绳子的长度记为k[0], k[1], ```, k[m]。请问它们的乘积可能的最大值是多少？例如，当绳子的长度为8时，我们把它剪成长度为2、3、3的三段，此时得到的乘积最大，是18.

解法一：动态规划。时间复杂度O(n^2)，空间复杂度O(n)。**灵活运用动态规划的关键是具备从上到下分析问题，并且从下到上解决问题的能力。**

解法二：贪婪算法。时间复杂度和空间复杂度均为O(n)。

```python
# -*- coding:utf-8 -*-
class Solution:
    def max_product(self, n):
        """动态规划"""
        if n < 2:
            return 0
        elif n == 2:
            return 1
        elif n == 3:
            return 2

        products = [0] * (n+1)
        products[0] = 0
        products[1] = 1
        products[2] = 2
        products[3] = 3

        for i in range(4, n+1):
            max_val = 0
            for j in range(1, i // 2 + 1):
                product = products[j] * products[i - j]
                if max_val < product:
                    max_val = product
                    products[i] = max_val
        print(products)
        return products[n]

    def max_product2(self, n):
        """贪婪算法
        数学上可以证明，当n>=5时，尽可能多的剪长度为3的绳子，且最后一段绳子如果是4的话，把它剪成2+2的两段。
        证明：当n >= 5时，下列不等式恒成立  3(n-1) >= 2(n-2) > n。
        因此，当n大于等于5时，尽可能剪成长度为3或2的小段，并且尽可能剪成长度为3的小段。并且当n=4时，剪成2+2比1+3更好。
        证毕！
        """
        if n < 2:
            return 0
        elif n == 2:
            return 1
        elif n == 3:
            return 2

        times_of_2 = 0
        times_of_3 = 0

        times_of_3 = n // 3
        if n % 3 == 1:
            times_of_3 -= 1
        times_of_2 = (n - times_of_3 * 3) / 2

        return int(3 ** times_of_3 * 2 ** times_of_2)

solution = Solution()
print(solution.max_product(80))
print(solution.max_product2(80))
```

# 面试题15：二进制中1的个数

>请实现一个函数，输入一个整数，输出该数二进制表示中1的个数。例如输入9,9的二进制表示是1001，则输出2.

## 位运算回顾

**位运算是把数字表示为二进制后，对每一位上0或1的运算，总共有六种：与(&)、或(|)、非(~)、异或(^)、左移(<<)和右移(>>)。**其中：
- 异或是**同0异1**   
- 左移运算符m << n表示把m左移n位，最左边的n位将会被抛弃，同时在最右边补上n个0  
- 右移运算符m >> n表示把m右移n位，最右边的n位将会被抛弃，如果数字是整数，则右移之后在最左边补n个0，如果数字是负数，则右移之后在最左边补n个1.

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        return sum([(n>>i & 1) for i in range(32)]) # 循环的次数等于整数二进制的位数，32位的整数需要循环32次

solution = Solution()
print(solution.NumberOf1(0x80000000))
```

**上面这种算法在C++里如果输入0x80000000会进入死循环(最后一直是0xFFFFFFFF)，但是Python里面不会。**

# 面试题16：数值的整数次方

>实现函数power(base, exponent)，其中base是double类型，exponent是整数类型。

解法一：需要考虑exponent为负数时取倒数的情况。还需要考虑当base=0时如果exponent为负数，要抛出异常。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0 and exponent < 0:
            raise ValueError
        result = 1
        for i in range(abs(exponent)):
            result *= base
        if exponent > 0:
            return result
        elif exponent == 0:
            return 1
        else:
            return 1. / result
```

解法二：指数分奇偶有不同的递归方法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0:
            return 0
        if exponent == 1:
            return base
        if exponent == 0:
            return 1
        flag = 0
        if exponent < 0:
            flag = -1
            exponent = -1 * exponent
        if flag == 0:
            if exponent % 2 == 0:
                return self.Power(base, exponent/2) * self.Power(base, exponent/2)
            if exponent % 2 == 1:
                return self.Power(base, (exponent-1)/2) * self.Power(base, (exponent-1)/2) * base
        else:
            if exponent % 2 == 0:
                return 1. / (self.Power(base, exponent/2) * self.Power(base, exponent/2))
            if exponent % 2 == 1:
                return 1. / (self.Power(base, (exponent-1)/2) * self.Power(base, (exponent-1)/2) * base)
```

# 面试题12：打印1到最大的n位数

## 打印1到最大的n位数

解法一：适合Python，不适合C++（大数溢出问题）。先算出最大范围，然后依次打印。不是面试官满意的算法。

解法三：递归。面试时用这个最好。**注意：打印函数要单独写一个，因为数字开头的0不应该被打印出来。**  

解法二：使用字符串模拟加法。但是代码太长 ，不推荐。

```python
def my_print(n):
    """python语言的优点，数字没有上限，可以直接打印"""
    max_val = 1
    for _ in range(n):
        max_val *= 10
    for i in range(1, max_val):
        print(i)
    return


def my_print2(n):
    """
    数字的全排列，用递归的方法实现
    """
    if n <= 0:
        return

    num = []
    #num_last = num
    #for i in range(10):
    #    num = [str(i)] + num_last
    #    print_recursive(n, num)
    print_recursive(n, num)
def print_recursive(n,num):
    if len(num) > n :
        return
    if len(num) == n:
        print_number(num)

    if len(num) <= n-1:
        num_last = num
        for i in range(10):
            num = num_last + [str(i)]
            print_recursive(n, num)

def print_number(num):
    is_begining0 = True
    n = len(num)
    for i in range(n):
        if is_begining0 and num[i] != '0':
            is_begining0 = False
        if not is_begining0:
            print(num[i], end="")
    print()

my_print(3)
my_print2(3)
```

## 补充题：大数相加

>定义一个函数，该函数可以实现任意两个整数的加法。

思路：任意两个整数的加法，因此是大数问题，需要用字符串表示或列表表示的加法。另外，还需要考虑负数的情况。

```python
def big_number_sum(s1, s2):
    """首先写出两个整数相加和相减的函数，然后根据两个数的正负号分四种类别计算"""
    if s1[0] != '-' and s2[0] != '-':
        big_positive_number_sum(s1, s2)
    elif s1[0] == '-' and s2[0] != '-':
        s1 = s1[1:]
        big_positive_number_minus(s2, s1)
    elif s1[0] != '-' and s2[0] == '-':
        s2 = s2[1:]
        big_positive_number_minus(s1, s2)
    else:
        s1 = s1[1:]
        s2 = s2[1:]
        print('-', end="")
        big_positive_number_sum(s1, s2)

def big_positive_number_sum(s1, s2):
    """两个大正数相加，首先对齐(补0)，然后按加法计算规则计算"""
    L1=[0]
    L2=[0]
    for i in range(0,len(s1)):
        L1.append(int(s1[i]))
    for i in range(0,len(s2)):
        L2.append(int(s2[i]))

    if(len(s1)>len(s2)):
        for i in range(len(s1)-len(s2)):
            L2.reverse()
            L2.append(0)
            L2.reverse()
    elif(len(s1)<=len(s2)):
        for i in range(len(s2)-len(s1)):
            L1.reverse()
            L1.append(0)
            L1.reverse()

    for i in range(len(L1)):
            L1[i]=L1[i]+L2[i]
    A=B=len(L1)-1

    while A>0:
        if((L1[A])/10)>=1:
            L1[A]=L1[A]%10
            L1[A-1]=L1[A-1]+1
        A-=1
    if L1[0]==0:
        for i in range(1,B+1):
            print(L1[i],end='')
    elif L1[0]!=0:
        for i in range(B+1):
            print(L1[i],end='')
    print()

def big_positive_number_minus(s1, s2):
    """减法：注意一点，如果s1减去s2得到的第一位的数字是-1，那么说明s2更大，先打印一个负号，然后调用big_positive_number_minus(s2, s1)"""
    L1=[0]
    L2=[0]
    for i in range(0,len(s1)):
        L1.append(int(s1[i]))
    for i in range(0,len(s2)):
        L2.append(int(s2[i]))

    if(len(s1)>len(s2)):
        for i in range(len(s1)-len(s2)):
            L2.reverse()
            L2.append(0)
            L2.reverse()
    elif(len(s1)<=len(s2)):
        for i in range(len(s2)-len(s1)):
            L1.reverse()
            L1.append(0)
            L1.reverse()

    for i in range(len(L1)):
            L1[i]=L1[i] - L2[i]
    A=B=len(L1)-1

    while A>0:
        if L1[A] < 0:
            L1[A]=L1[A] + 10
            L1[A-1]=L1[A-1] - 1
        A-=1
    if L1[0]==0:
        for i in range(1, B+1):
            print(L1[i],end='')
            if i == B:
                print()
    elif L1[0]!=0:
        print('-', end="")
        big_positive_number_minus(s2, s1)

big_number_sum('1323479817', '1372987318423498414')
big_number_sum('1323479817', '-1372987318423498414')
big_number_sum('-1323479817', '1372987318423498414')
big_number_sum('-1323479817', '-1372987318423498414')
print()
print(1323479817 + 1372987318423498414)
print(1323479817 - 1372987318423498414)
print(-1323479817 + 1372987318423498414)
print(-1323479817 - 1372987318423498414)
```

# 面试题18：删除链表的节点

## 题目一：在O(1)时间内删除链表节点。

>给定单向链表的头指针和一个节点指针(非结尾指针)，定义一个函数在O(1)时间内删除该节点。

解法分析：假设要删除的节点及其前后节点为h, i, j，常规的做法是从链表的头结点开始，直到遍历到h时发现它的下一个指针指向i，那么把h的下一个指针指向j，然后删除i。但是这么做时间复杂度为O(n)。之所以需要遍历一遍，是因为我们要找到待删除节点的前一个节点。但是，**更好的做法是不找前一个节点，把后一个节点的值和它的下一个节点的指针赋给当前待删除节点，然后删除后一个节点即可。**需要注意的是，如果要删除的节点是最后一个节点，那么还是需要遍历一遍。但是总体上来说，平均时间复杂度是O(1)。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val, node.next = node.next.val, node.next.next
```

## 题目二：删除链表中重复的节点

>在一个排序的链表中，如何删除重复的节点？

解法：注意全面地考虑问题

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        head1 = pHead.next
        if head1.val != pHead.val:
            pHead.next = self.deleteDuplication(head1)  # 当前节点和下一个节点不相同，所以当前节点不删除，递归找下一个节点
        else:
            while pHead.val == head1.val and head1.next is not None:
                head1 = head1.next
            if head1.val != pHead.val:
                pHead = self.deleteDuplication(head1)        # 当前节点和下一个节点不相同，但是pHead是重复节点，所以递归找当前节点
            else:
                return None
        return pHead
```

# 面试题19：正则表达式匹配

>请实现一个函数用来匹配包含'.'和'*'的正则表达式。模式中的字符'.'表示任何一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        """
        算法思路：需要特别注意的是：*表示前面的数字可以出现任意次，包括0！
        对于pattern下一个字符是*的情况，如果当前字符和pattern不匹配，
        那么pattern后移两个即可（这时*的意思是出现0次）；如果匹配，那么又要
        分三种情况；
        如果pattern下一个字符不是*，那么直接看当前字符是否匹配，匹配就s和
        pattern各往后一个
        
        递归的思路比用两个指针的方法要简单多了
        """
        if (len(s) == 0 and len(pattern) == 0):
            return True
        if (len(s) > 0 and len(pattern) == 0):
            return False
        
        if (len(pattern) > 1 and pattern[1] == '*'):
            if (len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.')):
                return (self.match(s, pattern[2:]) or self.match(s[1:], pattern[2:]) or self.match(s[1:], pattern))
            else:
                return self.match(s, pattern[2:])
        if (len(s) > 0 and (pattern[0] == '.' or pattern[0] == s[0])):
            return self.match(s[1:], pattern[1:])
        return False
```

# 面试题20：表示数值的字符串

>请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串“+100”、“5e2”、“-123”、“3.1416”及“-1E-16”都表示数值。但“12e”、“1a3.14”、“1.2.3”、
“+-5”及“12e+5.4”都不是。

解法：数值型字符串分为五部分[整数部分].[小数部分]e/E[指数部分]。其中整数部分可能有+、-号，如果有小数部分，那么整数部分也可以没有。小数部分也可以没有，如果有整数部分的话。指数部分也可能以+-号开头。

```python
# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        if s is None:
            return False
        flag, s = self.scanInteger(s)

        if s is not None and s[0] == '.':
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
            flag1, s = self.scanUnsignedInteger(s)
            flag = flag or flag1

        if s is not None and (s[0] == 'e' or s[0] == 'E'):
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
            flag2, s = self.scanInteger(s)
            flag = flag and flag2

        return flag and s is None

    def scanUnsignedInteger(self, s):
        flag = False
        while s is not None and '0' <= s[0] <= '9':
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
            flag = True
        return flag, s

    def scanInteger(self, s):
        if s is not None and (s[0] == '+' or s[0] == '-'):
            if len(s) > 1:
                s = s[1:]
            else:
                s = None
        return self.scanUnsignedInteger(s)


print(Solution().isNumeric("12."))
print(Solution().isNumeric("12.e"))
print(Solution().isNumeric("12.1e-10"))
```

也可以直接用python的float函数和try except.

```python
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        try :
            p = float(s)
            return True
        except:
            return False
```

# 面试题21：调整数组顺序使奇数位于偶数前面

>输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分。

解法分析：如果从前到后扫描数组，遇到一个偶数就把它拿出来，后面的数字全部前移，然后再把这个偶数放到最后一个空出来的位置。这么做时间复杂
度是O(n^2)，不好。另一种做法是设计两个数组，从前往后依次扫描原数组，如果是奇数则append到新数组1中，如果是偶数则append到新数组2中。最后
新数组1和新数组2合并。时间复杂度是O(n)，但是空间复杂度也是O(n)。如果空间复杂度要求不高那么这个方法就可以。

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        even_list = []
        odd_list = []
        for value in array:
            if value % 2 == 1:
                odd_list.append(value)
            else:
                even_list.append(value)
        return odd_list + even_list
```

解法2：双端队列。从前向后扫描数组中的元素，如果有偶数，则append进双端队列；从后向前扫描数组中的元素，如果是奇数，则appendleft进双端队列。
注意需要两个方向分别扫描，因为必须保证奇数和偶数的相对位置不变。

```python
# -*- coding:utf-8 -*-
from collections import deque
class Solution:
    def reOrderArray(self, array):
        # write code here
        odd = deque()
        x = len(array)
        for i in range(x):
            if array[i] % 2 == 0:
                odd.append(array[i])
            if array[x - i - 1] % 2 == 1:
                odd.appendleft(array[x - i - 1])
        return list(odd)
```

解法3：数组内互换的方式，时间复杂度高O(n^2)，但是空间复杂度O(1)

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        count = 0
        for i in range(0,len(array)):
            for j in range(len(array)-1,i,-1):
                if array[j-1]%2 ==0 and array[j]%2==1:
                    array[j-1], array[j] = array[j], array[j-1]
        return array
```

# 链表中倒数第k个节点

>输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾结点是倒数第一个节点。例如，一个链表有6个节点，
从头节点开始，它们的值依次是1、2、3、4、5、6,。这个链表的倒数第三个节点值为4.

解法1：倒数第k个节点就是正数第 n-k+1 个。第一次遍历，得到所有节点数n的值，第二次遍历，找到第n-k+1个节点，输出其值。

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
        n = 1 # total node number
        node = head
        while node.next is not None:
            n += 1
            node = node.next
        
        if k > n:
            return None
        for i in range(n-k+1-1):
            head = head.next
        return head
```

解法2：如果只能遍历一次，怎么做？双指针。首先第一个指针从头开始遍历k-1步，第二个指针不变，此时两个指针的距离是k-1。那么后面同时向后移动
两个指针，如果第一个指针到达最后一个位置，那么第二个指针对应的节点就是我们找的。

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
        if k == 0:
            return None
        node = head
        for i in range(k-1):
            if node.next is not None:
                node = node.next
            else:
                return None

        while node.next is not None:
            node = node.next
            head = head.next
        return head
```

**这道题的代码要注意鲁棒性，如果特殊情况没有考虑导致代码崩溃，很难拿到offer**

拓展：**求链表的中间节点，如果总结点数为奇数就是中间那个，如果是偶数就返回中间两个节点的任意一个**。同样可以用遍历两次的方法或者两个指针
的方法。两个指针的方法是，设置第一个指针每次走一步，第二个指针每次走两步，这样当第二个指针的下下个节点为空时，它已经到最后一个节点或者倒
数第二个节点，此时第一个指针对应的节点就是中间节点

# 面试题23：链表中环的入口节点

>如果一个链表中有环，如何找到环的入口节点？

解法：

- 首先是判断链表中有环。具体做法是两个速度不一样的指针，速度分别是一次一步和一次两步，如果走的快的指针追上了走得慢的指针，那么说明
链表有环；如果走的快的指针到链表的结尾（指针对应节点是None）也没有追上，那么就没有环。

- 然后是找到环的入口节点。找到链表中环中节点的个数n，然后让第一个节点先走n步，然后两个节点以相同的速度（一次一步）向前移动，两者相遇的
节点就是环的入口。

- 找到n的方法。如果链表有环，两者相遇的节点肯定在环内，然后从这个节点出发，一边计数一边移动，再次回到这个节点就可以知道环中的节点数量n。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        
        if pHead is None or pHead.next is None or pHead.next.next is None:
            return None
        if pHead.next == pHead:
            return pHead
        node1 = pHead
        node2 = pHead.next.next
        
        while node1 != node2:
            if node2.next.next is not None:
                node1 = node1.next
                node2 = node2.next.next
            else:
                return None
        num = 1
        node = node1
        node = node1.next
        while node1 != node:
            node = node.next
            num += 1
        node3 = pHead
        node4 = pHead
        
        for i in range(num):
            node4 = node4.next
        
        while node3 != node4:
            node3 = node3.next
            node4 = node4.next
        
        return node3
```

# 面试题24：反转链表

>定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, head):
        # write code here
        if head is None or head.next is None:
            return head
        if head.next.next is None:
            node = head.next
            node.next = head
            head.next = None
            return node
        node_b, node, node_n = None, head, head.next
        while node.next is not None:# 离开循环体时node是最后一个结点，每次只调整一个指针
            node_n = node.next
            node.next = node_b
            node_b = node
            node = node_n
        node.next = node_b    
        return node
```

# 面试题25：合并两个排序的链表

>输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

解法：高频面试题，用递归做比较好。注意代码的鲁棒性，提前考虑空链表的情况。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1 is None:
            return pHead2
        elif pHead2 is None:
            return pHead1
        pMergedHead = ListNode(None)
        
        if pHead1.val < pHead2.val:
            pMergedHead = pHead1
            pMergedHead.next = self.Merge(pHead1.next, pHead2)
        else:
            pMergedHead = pHead2
            pMergedHead.next = self.Merge(pHead1, pHead2.next)
        return pMergedHead
```

# 面试题26：树的子结构

>输入两棵二叉树A和B，判断B是不是A的子结构。

**和链表相比，树的指针操作更多也更复杂，因此与树相关的问题通常会比链表更难。**

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.lis = []
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if pRoot1 is None or pRoot2 is None:
            return False
        flag = False
        self.front_recursive(pRoot1)
        for node in self.lis:
            if node.val == pRoot2.val:
                flag = flag or self.compare(node, pRoot2)
        return flag
        
    def front_recursive(self, root):
        if root is None:
            return 
        self.lis.append(root)
        self.front_recursive(root.left)
        self.front_recursive(root.right)
        
    def compare(self, root1, root2):
        if root2 is None:
            return True
        if root1 is None and root2 is not None:
            return False
        flag = True
        if root1.val == root2.val:
            flag = True
        else:
            flag = False
        return flag and self.compare(root1.left, root2.left) and self.compare(root1.right, root2.right)
```
上面的代码需要一个单独的列表存放所有节点，不是最好的方法。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        return self.is_subtree(pRoot1, pRoot2) or self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)
     
    def is_subtree(self, A, B):
        if not B:
            return True
        if not A or A.val != B.val:
            return False
        return self.is_subtree(A.left,B.left) and self.is_subtree(A.right, B.right)
```

# 面试题27：二叉树的镜像

>请完成一个函数，输入一棵二叉树，该函数输出它的镜像。

解法：首先注意的是改变原树，而不是返回一棵新的树。具体的做法就是交换左右子节点的值，然后用递归的方法遍历。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        """遍历一棵树的同时交换左右节点的值"""
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
```

# 面试题28：对称的二叉树。

>请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

解法：**注意并不是保证所有节点的左右子节点相等就行**。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        res = self.is_sym(pRoot, pRoot)
        return res
    
    def is_sym(self, root1, root2):  
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        if root1.val != root2.val:
            return False
        return self.is_sym(root1.left, root2.right) and self.is_sym(root1.right, root2.left)
```

# 面试题29：顺时针打印矩阵

解法：要提高程序的鲁棒性，一定要（1）提前进行特殊情况处理，比如只有一个元素的矩阵；比如只有一行的矩阵（因为输入是二维列表，虽然只有一行
但是提取元素时也得两次索引） （2）想好打印一圈的四个点的两个索引是什么，打印每一行每一列的起点和终点是什么

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        """
        [o, o]         ...            [o, cols-o-1]
              .                                        .
              .                                        .
              .                                        .
        [rows-o-1, o]  ...            [rows-o-1, cols-o-1]
        """
        res=[]
        rows=len(matrix)
        cols=len(matrix[0])
        if rows==1 and cols==1:
            res=[matrix[0][0]]
            return res
        if cols == 1:
            res = [i[0] for i in matrix]
            return res
        for o in range((min(rows,cols)+1)//2):
            [res.append(matrix[o][i]) for i in range(o,cols-o)]
            [res.append(matrix[j][cols-1-o]) for j in range(o+1,rows-o)]
            [res.append(matrix[rows-1-o][k]) for k in range(cols-1-o-1,o-1,-1) if rows-1-o != o]
            [res.append(matrix[l][o]) for l in range(rows-1-o-1,o,-1) if cols-0-1 != o]
        return res
```

# 面试题30：包含min函数的栈

>定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的min函数。在该栈中，调用min、push和pop的时间复杂度都是O(1)。

解法：另外定义一个辅助栈，每次对栈做push时都比较一下当前值和辅助栈顶端值的大小，如果push的数小则把该数也push进辅助栈里。
对栈pop时检查是否是当前最小值（辅助栈的栈顶元素），如果是，则两个栈都要pop

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
        self.stack2 = [] # 辅助栈
    def push(self, node):
        # write code here
        self.stack.append(node)
        if not self.stack2:
            self.stack2.append(node)
        else:
            if self.stack2[-1] > node:
                self.stack2.append(node)
        return 
    def pop(self):
        # write code here
        if not self.stack:
            return
        if self.stack[-1] == self.stack2[-1]:
            self.stack2.pop(-1)
        return self.stack.pop(-1)
    def top(self):
        # write code here
        if not self.stack:
            return 
        return self.stack[-1]
    def min(self):
        # write code here
        return self.stack2[-1]
```

# 面试题31：栈的压入、弹出序列

>输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
（注意：这两个序列的长度是相等的）

解法：先定义一个空的栈。遍历pushV，如果值和popV的0号元素相等，则popV.pop(0)；如果不相等，则把元素append进栈；
遍历结束后遍历新得到的栈，如果栈的-1元素和popV的0元素相等，就分别pop掉；否者返回错误；遍历完(代表没有返回错误)就返回正确。

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        stack = []
        if not pushV and not popV:
            return True
        for val in pushV:
            if val == popV[0]:
                popV.pop(0)
            else:
                stack.append(val)
        while stack:
            if stack[-1] != popV[0]:
                return False
            else:
                stack.pop(-1)
                popV.pop(0)
        return True
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
    
        def print_binary_tree4(self):
        """之字形打印二叉树，在分行打印的基础上添加一个记录本层顺序的值，之前打印的地方换成压到一个列表"""
        if not self.root:
            return
        res = [] # 存放层次遍历序列
        queue = [self.root]
        to_be_printed = 1
        next_level_num = 0
        level_nums = [to_be_printed]  # 存放每一层的节点数

        while queue:
            node = queue.pop(0)
            if to_be_printed > 0:
                res.append(node.elem)
                to_be_printed -= 1
            else:
                to_be_printed = next_level_num
                level_nums.append(next_level_num)
                next_level_num = 0
                res.append(node.elem)
                to_be_printed -= 1
            if node.lchild:
                queue.append(node.lchild)
                next_level_num += 1
            if node.rchild:
                queue.append(node.rchild)
                next_level_num += 1

        for i, nums in enumerate(level_nums):
            if i % 2 == 0:
                for _ in range(nums):
                    print(res.pop(0), end=" ")
                print()
            else:
                temp = []
                for _ in range(nums):
                    temp.append(res.pop(0))
                for j in range(len(temp)-1, -1, -1):
                    print(temp[j], end=" ")
                print()
        return



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
        res = []
        left = self.FindPath(root.left, expectNumber-root.val)
        right = self.FindPath(root.right, expectNumber-root.val)
        
        for i in left + right:
            res.append([root.val] + i)
        return res
```

# 面试题35：复杂链表的复制

>输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

解法：难点在于random指针的复制。举个例子，原链表中N的random指向S，那么新的链表中N'的random要指向S'。**使用两个哈希表。** 时间复杂度O(n)，空间复杂度O(n)


```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
         
        head = pHead
        p_head = None
        new_head = None
         
        random_dic = {}
        old_new_dic = {}
         
        while head:
            node = RandomListNode(head.label)
            node.random = head.random
            old_new_dic[id(head)] = id(node)
            random_dic[id(node)] = node
            head = head.next
             
            if new_head:
                new_head.next = node
                new_head = new_head.next
            else:
                new_head = node
                p_head = node
                 
        new_head = p_head
        while new_head:
            if new_head.random != None:
                new_head.random = random_dic[old_new_dic[id(new_head.random)]]
            new_head = new_head.next
        return p_head
```
面试题36：二叉搜索树和双向链表

>输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

```python
#-*- coding:utf-8 -*-
#class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        """
        难点在于不能创建任何新的节点，只能调整节点指针的指向
        递归地处理左子树、根节点和右子树的指针
        """
        if not pRootOfTree:
            return None
        if not pRootOfTree.left and not pRootOfTree.right:
            return pRootOfTree

        # 处理左子树
        self.Convert(pRootOfTree.left)
        left = pRootOfTree.left

        # 连接左子树的最大节点和跟节点
        if left:
            while left.right:
                left = left.right
            pRootOfTree.left, left.right = left, pRootOfTree

        # 处理右子树
        self.Convert(pRootOfTree.right)
        right = pRootOfTree.right

        # 连接右子树的最小节点和根节点
        if right:
            while right.left:
                right = right.left
            pRootOfTree.right, right.left = right, pRootOfTree

        # 找到最小的节点
        while pRootOfTree.left:
            pRootOfTree = pRootOfTree.left
        return  pRootOfTree
```

**注意，IDE和牛客相互复制有时会有缩进的问题**

# 面试题37：序列化二叉树

>请实现两个函数，分别用来序列化和反序列化二叉树。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    flag = -1
    def Serialize(self, root):
        # write code here
        if not root:
            return "#"
        return str(root.val) + ',' + self.Serialize(root.left) + ',' + self.Serialize(root.right)

    def Deserialize(self, s):
        # write code here
        self.flag += 1

        l = s.split(",")
        if self.flag >= len(s):  # 为什么终止条件是len(s)？
            return None

        root = None
        if l[self.flag] != "#":
            root = TreeNode(int(l[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        return root
```

# 面试题：平衡二叉树

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        """
        它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，
        并且左右两个子树都是一棵平衡二叉树
        """
        if not pRoot:
            return True
        if abs(self.TreeDepth(pRoot.left) - self.TreeDepth(pRoot.right)) > 1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
    
    def TreeDepth(self, root):
        if not root:
            return 0
        left = self.TreeDepth(root.left)
        right = self.TreeDepth(root.right)
        
        return max(left+1, right+1)
```

# 面试题38：字符串的排列

>输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        """
        把复杂问题分解为小问题，递归
        把整个字符串分成两部分：第0个字符和后面所有的字符，
        第一步求所有可能在第一个位置的字符，
        第二步固定第一个字符，求后面所有字符的排列
        """
        if not ss:
            return []
        res = []
        self.per(ss, res, "")
        unique = list(set(res))
        return sorted(unique)
        
    def per(self, ss, res, path):
        if ss == "":
            res.append(path)
        for i in range(len(ss)):
            self.per(ss[:i] + ss[i+1:], res, path+ss[i])# 注意对于一个长度为n的字符串a,a[n]会报错，但是a[n:]是""   
```
```python
# -*- coding:utf-8 -*-
import itertools
class Solution:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return []
        return sorted(list(set(map(''.join, itertools.permutations(ss)))))
```

# 面试题39：数组中出现次数超过一半的数字

题目：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。  

解法1： 从数组的特点出发，只需要在遍历数组时保存两个数，一个是数组中的一个数，一个是次数。当遍历到下一个数字时，如果次数为0，则把当前的数字赋值给保存的数字，并把次数置为1，如果下一个数字和之前保存的数字相同，则次数加一，否则次数减一。
```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        """
        次数超过一半
        可以通过一个flag和临时的value来记录遍历一次的结果，
        如果当前的flag是0，那么直接赋值给value，同时flag+=1
        如果当前flag不是0，那么比较value和当前值，相等则flag+=1，
        不相等则flag-=1
        最后如果flag>0并且value的次数大于一半的长度，则返回value的值；
        否则返回0
        """
        if not numbers:
            return 0
        value = 0
        flag = 0
        for i in numbers:
            if flag == 0:
                value = i
                flag = 1
            elif i == value:
                flag += 1
            else:
                flag -= 1
        if flag > 0 and self.check_more_than_half(numbers, value):
            return value
        else:
            return 0
    def check_more_than_half(self, numbers, value):
        times = 0
        for i in numbers:
            if i == value:
                times += 1
        if times > len(numbers) / 2:
            return True
        else:
            return False
            
```



# 面试题40：最小的k个数

**这个题绝对是高频题，很多面经里面都提到了** 

解法1：:维护一个小顶堆，时间复杂度O((n-k)logk + klogk) = O(nlogk)。适合处理海量数据

```python
# -*- coding:utf-8 -*-
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        """
        由于默认的是小顶堆，因此先乘以-1，求最大的K个数，
        维护一个小顶堆，
        最后得到的结果再乘以-1，然后快排一下，返回
        """
        if len(tinput) < k or k == 0:   # 不要忘了k=0的时候
            return []
        if len(tinput) == k:
            return self.quick_sort(tinput)  # 注意这里要排序不要忘了
        tinput = [-1 * x for x in tinput] # 下面求最大的k个数，每个数和小顶堆的顶元素比较
        heap = tinput[:k]
        heapq.heapify(heap)
        for x in tinput[k:]:
            if x > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, x)
        return self.quick_sort([-1 * val for val in heap])
    def quick_sort(self, nums):
        if len(nums) <= 1:
            return nums
        mid_val = nums[0]
        below = [x for x in nums[1:] if x <= mid_val]
        above = [x for x in nums[1:] if x > mid_val]
        return self.quick_sort(below) + [mid_val] + self.quick_sort(above)
```
解法2：最小堆，时间复杂度O(klogn)。每次用堆排序求出最小值，保存然后pop，重复k次。

```python
# -*- coding:utf-8 -*-
import heapq
 
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or not k or k > len(tinput):
            return []
        heapq.heapify(tinput)
        return [heapq.heappop(tinput) for _ in range(k)]
```

解法2：基于partition函数。只有当可以改变数组时用，时间复杂度为O(n)。


# 面试题41：数据流中的中位数

>如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

解法1：使用数组，直接插入O(1)，排序然后求中位数O(nlogn).
```python
# -*- coding:utf-8 -*-
class Solution:
    arr = []

    def Insert(self, num):
        # write code here
        self.arr.append(num)

    def GetMedian(self, data):
        # write code here
        self.arr.sort()
        length = len(self.arr)
        if length == 0:
            return None
        elif length == 1:
            return self.arr[0]
        elif length % 2 == 1:
            return self.arr[(length - 1) // 2]
        else:
            return (self.arr[(length - 1) // 2] + self.arr[length // 2]) / 2.
```
解法2：还是使用数组，直接插入O(1)，使用O(n)的时间复杂度求中位数。**使用partition的方法可以用O(n)的时间复杂度从一个数组中找到第k大的元素**

**从数组中找第k小的元素**
```python
def partition(num, low, high):
    pivot = num[low]
    while low < high:
        while low < high and num[high] > pivot:
            high -= 1
        while low < high and num[low] < pivot:
            low += 1
        num[low], num[high] = num[high], num[low]
    num[low] = pivot
    return low

def quick_sort(num, low, high):
    if low < high:
        location = partition(num, low, high)
        quick_sort(num, low, location - 1)
        quick_sort(num, location + 1, high)
    return num
def find_kth(num, low, high, k): # 找到从小到大第k个数（序号从0开始）
    index = partition(num, low, high)
    if index == k:
        return num[index]
    if index < k:
        return find_kth(num, index+1, high, k)
    else:
        return find_kth(num, low, index - 1, k)

num = [4,3,1,5,6,2]
print(quick_sort(num, 0, len(num)-1))
print(find_kth(num, 0, len(num)-1, 3))
```

```python
# -*- coding:utf-8 -*-
class Solution:
    arr = []

    def Insert(self, num):
        # write code here
        self.arr.append(num)

    def GetMedian(self):  # 牛客网上这里需要多写一个参数，但是无意义
        # write code here
        l = len(self.arr)
        if l == 0:
            return None
        if l == 1:
            return self.arr[0]
        if l % 2 == 0:
            return (self.find_kth(self.arr, 0, l-1, (l-1)//2) + self.find_kth(self.arr, 0, l-1, l//2))/ 2.
        if l % 2 == 1:
            return self.find_kth(self.arr, 0, l-1, (l-1)//2)

    def partition(self, num, low, high):
        pivot = num[low]
        while low < high:
            while low < high and num[high] > pivot:
                high -= 1
            while low < high and num[low] < pivot:
                low += 1
            num[low], num[high] = num[high], num[low]
        num[low] = pivot
        return low
    def find_kth(self, num, low, high, k):
        index = self.partition(num, low, high)
        if index == k:
            return num[index]
        if index < k:
            return self.find_kth(num, index+1, high, k)
        if index > k:
            return self.find_kth(num, low, index-1, k)


s = Solution()
s.Insert(5)
print(s.GetMedian())
s.Insert(2)
print(s.GetMedian())
s.Insert(3)
print(s.GetMedian())
s.Insert(4)
print(s.GetMedian())
s.Insert(1)
print(s.GetMedian())
s.Insert(6)
print(s.GetMedian())
s.Insert(7)
print(s.GetMedian())
```

解法3：最大堆和最小堆。插入O(logn)，查询中位数O(1)。

# 面试题42：连续子数组的最大和

>HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,
常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,
并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，
返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

分析：枚举法比较直观，对于一个长度为n的数组，总共有n(n+1)/2个子数组。计算出所有子数组的和，最快也要O(n^2)时间。这个方法是通不过面试的。

解法：**动态规划。如果以f(i)表示第i个数字结尾的子数组的最大和，那么我们最终求的是max(f(i))。如果i=0或者f(i-1)<0，那么f(i)就等于第i个数；
如果i不等于0且f(i-1)>0，那么f(i)=f(i-1) + 第i个数**


```python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if not array:
            return None
        if len(array) == 1:
            return array[0]
        max_sum, cur_sum = -0xffffff, 0
        for i in array:
            if cur_sum <= 0:
                cur_sum = i
            else:
                cur_sum += i
            if max_sum <= cur_sum:
                max_sum = cur_sum
        return max_sum
```

# 面试题43:1~n整数中1出现的次数

>求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13
因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数
（从1 到 n 中1出现的次数）。

解法1：遍历一次，对每个遍历到的数，求出1的个数然后累加到res上。最后返回res。时间复杂度O(nlogn)，效率比较低。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        res = 0
        for i in range(1, n+1):
            res += self.number_of_1(i)
        return res
    
    def number_of_1(self, num):
        res = 0
        while num:
            if num % 10 == 1:
                res += 1
            num /= 10
        return res
```

解法2：找规律，循环次数和位数相同，一个数字n有O(logn)位，所以时间复杂度O(logn)。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        ones, m = 0, 1
        while m <= n:
            ones += (n // m + 8) // 10 * m + (n // m % 10 == 1) * (n % m + 1)
            m *= 10
        return ones
```

面试题44：数字序列中某一位的数字

>数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从0开始计数）是5，第13位是1，第19位是4，等等。
请写一个函数，求任意第n位对应的数字。

解法1：从0开始逐一枚举数字，并把该数字的长度加进来，如果总长度大于等于n，那么第n位对应的数字就在这个数的某一位。
```python
def digit_at_index(n):
    if n < 0:
        return None
    if n == 0:
        return 0
    i = 0
    total_len = 0
    while total_len < n:
        i += 1
        total_len += len(str(i))

    print(i, total_len, n, end=" ")
    return int(str(i)[-1 - (total_len - n)])  # 注意这里的索引

print(digit_at_index(0))  # 0
print(digit_at_index(5))  # 5
print(digit_at_index(10)) # 1
print(digit_at_index(13)) # 1
print(digit_at_index(19)) # 4
```

# 面试题45：把数组排成最小的数

>输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

```python
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if len(numbers) == 0:
            return ""

        numbers_str = list(map(str, numbers))
        res = self.sort(numbers_str)
        print(numbers)
        print(numbers_str)
        print(res)
        temp = "".join(res)
        #print(temp)
        return int(temp)

    def sort(self, s):
        if len(s) == 0 or len(s) == 1:
            return s
        temp = s[0]
        bf = []
        af = []
        for i in range(1, len(s)):
            if temp + s[i] < s[i] + temp:
                af.append(s[i])
            else:
                bf.append(s[i])
        return self.sort(bf) + [temp] + self.sort(af)

s = Solution()
print(s.PrintMinNumber([3,32,321]))
```

面试题46：把数字翻译成字符串

> 给定一个数字，我们按照如下规则把它翻译为字符串；0翻译成“a”，1翻译成“b”,…
11翻译成“l”,…,25翻译成“z”.一个数字可能有多个翻译。例如，12258有5种不同的翻译，
分别是“bccfi” “bwfi” “bczi” “mcfi” “mzi” 请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

分析：递归的思路。定义f(i)表示从第i位数字开始的不同翻译数目，那么f(i) = f(i+1) + g(i, i+1)*f(i+2)。如果第i位和第i+1位两位数字
拼起来的数字在10~25范围内，函数g(i, i+1)的值为1，否则为0。**但是递归重复子问题，效率不高，因此要写基于循环的代码**

```python
class Solution:
    def numDecodings(self, s):
        pp, p = 1, int(s[0] != '0')
        for i in range(1, len(s)):
            pp, p = p, pp * (0 <= int(s[i-1:i+1]) <= 25) + p
        return p

s = Solution()
print(s.numDecodings("12258"))
print(s.numDecodings("12"))
```

# 面试题47：礼物的最大价值

>在一个mxn的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或
向下移动一格，知道到达棋盘的右下角。给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？

解法：**典型的动态规划问题**。先用递归分析，定义一个函数f(i,j)表示到达坐标为(i,j)的格子时能拿到的礼物总和的最大值。
f(i,j) = max(f(i-1, j), f(i, j-1)) + gift[i, j]。gift[i, j]表示坐标为(i,j)的格子里礼物的价值。由于递归的代码有大量重复的计算，
因此要写基于循环的代码。**使用一个辅助的二维数组，数组中坐标(i,j)的元素表示到达坐标(i,j)的格子时能拿到的礼物价值总和的最大值**

```python
def get_max_value(values, rows, cols): # value是一维的
    if len(values) == 0:
        return 0

    max_values = [[0] * cols] * rows

    for i in range(0, rows):
        for j in range(0, cols):
            left = 0
            up = 0
            if i > 0:
                up = max_values[i-1][j]
            if j > 0:
                left = max_values[i][j-1]
            max_values[i][j] = max(up, left) + values[i * cols + j]

    max_val = max_values[rows - 1][cols - 1]
    print(max_val)
    return max_val
get_max_value([1,10,3,8,12,2,9,6,5,7,4,11,3,7,16,5], 4, 4)
```

解法二：优化上面的代码。由于每次计算时只需要左边和上边的max_value，那么不需要存储一个二维数组，只需要一个一维数据（长度等于列数）
即可。

```python
def get_max_value(values, rows, cols): # value是一维的
    if len(values) == 0:
        return 0

    max_values = [0] * cols

    for i in range(0, rows):
        for j in range(0, cols):
            left = 0
            up = 0
            if i > 0:
                up = max_values[j]  # 这个在上一行的循环中已经得到了
            if j > 0:
                left = max_values[j-1]  # 这个在上一列的循环中得到的
            max_values[j] = max(up, left) + values[i * cols + j]

    max_val = max_values[-1]
    print(max_val)
    return max_val
get_max_value([1,10,3,8,12,2,9,6,5,7,4,11,3,7,16,5], 4, 4)
```

# 面试题48：最长不含重复字符的子字符串

>给定一个字符串，找到最长的子字符串的长度而不重复字符

解法：暴力求解法复杂度分析。一个长度为n的数组，子数组的数量为O(n^2)，即n + (n-1) + ... + 1，然后对于每个子数组要判断是否有重复
字符，O(n)，总体上是O(n^3)的时间复杂度。使用动态规划法提高效率。定义函数f(i)表示第i个字符结尾的不包含重复字符的子字符串的长度。
分类讨论f(i)的计算：如果第i个字符之前没有出现过，那么f(i) = f(i-1) + 1；如果第i个字符之前出现过，那么计算i和第i个字符上次出现在
字符串中的位置的距离d。如果d <= f(i-1)，那么说明第i个字符上次出现的位置是f(i-1)对应的最长子字符串中，且两个所夹的子字符串没有
其他重复的字符了。因此f(i) = d；如果d > f(i-1)，那么说明第i个字符上次出现在f(i-1)对应的最长子字符串之前，因此仍然有
f(i) = f(i-1) + 1。

```python
 def longest_substring_without_duplication(s):
    if len(s) <= 1:
        return len(s)
    position = [-1] * 26
    cur_len = 0
    max_len = 0

    for i, val in enumerate(s):
        d = i - position[ord(val) - ord('a')]
        if d > cur_len: # 第i个字符之前没有出现过和第i个字符之前出现过且
                                                          # d > f(i-1)(i和第i个字符上次出现在字符串中的位置的距离d)
            cur_len += 1
        else:
            if cur_len > max_len:
                max_len = cur_len
            cur_len = d
        position[ord(val) - ord('a')] = i

    if cur_len > max_len:
        max_len = cur_len
    print(max_len)
    return max_len

longest_substring_without_duplication('arabcacfr')
```

# 面试题49：丑数

>把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。
求按从小到大的顺序的第N个丑数。

解法1：逐个判断每个整数是不是丑数，知道第n个截止。缺点：每个整数都需要计算，即使一个数不是丑数也需要对它进行取余数和除法操作。因此算法
的时间效率不高。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        count = 1
        i = 1
        while count < index:
            i += 1
            if self.is_ugly(i):
                count += 1
        return i
    def is_ugly(self, num):
        while(num % 2 == 0):
            num = num // 2
        while(num % 3 == 0):
            num = num // 3
        while(num % 5 == 0):
            num = num // 5
        return num == 1
s = Solution()
print(s.GetUglyNumber_Solution(6))
print(s.GetUglyNumber_Solution(1500))
```

解法2：创建数组保存已经找到的丑数，用空间换时间。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index <= 0:
            return 0
        ugly_list = [1]
        index_2 = 0
        index_3 = 0
        index_5 = 0
        for i in range(index - 1):
            #print(index_2, index_3, index_5)
            new_ugly = min(ugly_list[index_2] * 2, ugly_list[index_3] * 3, ugly_list[index_5] * 5)
            ugly_list.append(new_ugly)
            if new_ugly % 2 == 0:
                index_2 += 1
            if new_ugly % 3 == 0:
                index_3 += 1
            if new_ugly % 5 == 0:
                index_5 += 1

        return ugly_list[-1]


s = Solution()
print(s.GetUglyNumber_Solution(20))
print(s.GetUglyNumber_Solution(1500)) # 859963392
```

# 面试题50：第一个只出现一次的字符

## 题目一：字符串中第一个只出现一次的字符

>在字符串中找出第一个只出现一次的字符。如输入"abaccdeff"，则输出'b'。

解法：哈希表法，两次遍历，时间复杂度O(n)

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if not s:
            return -1
        dic = {}
        for val in s:
            if val not in dic:
                dic[val] = 1
            else:
                dic[val] += 1
        for i, val in enumerate(s):
            if dic[val] == 1:
                return i
```

## 题目二：字符流中第一个只出现一次的字符

>请实现一个函数，用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"；当从
该字符流中读出前6个字符"google"时，第一个只出现一次的字符是"l"。

解法：同样是哈希表。


# 面试题51：数组中的逆序对

>在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。
并将P对1000000007取模的结果输出。 即输出P%1000000007

解法：两次循环，外层循环遍历每个数字，内层循环遍历这个数字后面的数字。时间复杂度O(n^2)，这个代码写出来基本上面试就凉了。


解法2：归并排序。

```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        global count
        count=0
        def A(array):
            global count
            if len(array)<=1:
                return array
            k=int(len(array)/2)
            left=A(array[:k])
            right=A(array[k:])
            l=0
            r=0
            result=[]
            while l<len(left) and r<len(right):
                if left[l]<right[r]:
                    result.append(left[l])
                    l+=1
                else:
                    result.append(right[r])
                    r+=1
                    count+=len(left)-l
            result+=left[l:]
            result+=right[r:]
            return result
        A(data)
        return count%1000000007
```

# 面试题52：两个链表中的第一个公共节点

>输入两个链表，找出它们的第一个公共节点。

解法一：暴力法。在第一个链表上遍历每个节点，然后在第二个链表上遍历每个节点，如果两个节点相同，则返回该节点。时间复杂度O(nm)。

解法二：空间换时间。由于是单链表，所以从第一个公共节点开始，后面的部分是重合的。因此从每个链表的最后一个节点开始，往前遍历，如果遇到最后一个
相同的节点，那么该节点就是第一个公共节点。具体实现上是使用两个栈，存储每个链表的所有节点，然后不断比较栈顶元素。时间复杂度O(n+m)，
空间复杂度O(n+m)。

解法三：解法二之所以要用栈，是因为我们要从后面往前遍历；之所以要从后往前遍历，是因为两个链表的节点数可能不一样。因此可以先遍历一下两个
链表，得到各自的节点数。然后让长的那个链表的head指针先走节点数差值次。然后两个指针一起往后走，第一次遇到相同的节点就是第一个公共节点。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if pHead1 is pHead2:
            return pHead1
        num1 = 0
        num2 = 0
        temp = pHead1
        while temp is not None:
            num1 += 1
            temp = temp.next
        temp = pHead2
        while temp is not None:
            num2 += 1
            temp = temp.next
        if num1 > num2:
            for _ in range(num1 - num2):
                pHead1 = pHead1.next
            while True:
                if pHead1 is pHead2:
                    return pHead1
                pHead1 = pHead1.next
                pHead2 = pHead2.next
```

# 面试题53：在排序数组中查找数字

## 题目一：数字在排序数组中出现的次数

>统计一个数字在排序数组中出现的次数。例如，输入排序数组{1,2,3,3,3,3,4,5}和数字3，由于3在这个数组中出现了4次，因此输出4.

解法1：顺序遍历统计次数，时间复杂度O(n)，不好。

解法2：因为是排序数组中的查找，因此使用二分查找。找到某个目标数字，然后在其左右两边扫描统计该数字的次数。因为要查找的数字出现的
次数可能是n，因此时间复杂度O(n)，和上一个一样，不好。

解法3：二分查找，去上一个方法的区别是直接查找第一个目标数字和最后一个目标数字。时间复杂度O(logn)。如果middle_value小于目标数字，那么第一个目标数字
在右半段；如果middle_value大于目标数字，那么第一个目标数字在左半段；如果等于目标数字，那么查看middle_value的前一个数字是否等于
目标数字，如果不等于，则middle_value是第一个目标数字，如果等于，则middle_value在右半段。上面的分析，目标数字在左半段，则
把right指针移到middle_value的位置；反之把left指针移到middle_value的位置。查找最后一个数字同理。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if len(data) == 0:
            return 0
        if len(data) == 1 and data[0] == k:
            return 1
        exist = 0
        first = 0
        last = 0

        left = 0
        right = len(data) - 1
        # find first
        while (right - left) > 1:
            middle = (left + right) // 2
            if data[middle] == k:
                exist = 1
                if middle == 0:
                    first = 0
                    break
                else:
                    if data[middle-1] == k:
                        right = middle
                        first = middle - 1
                    else:
                        first = middle
                        break
            elif data[middle] < k:
                left = middle
            else:
                right = middle
        print(left,right)
        # find last
        left = 0
        right = len(data) - 1
        while (right - left) > 1:
            middle = (left + right) // 2
            if data[middle] == k:
                exist = 1
                if middle == len(data) - 1:
                    last = len(data) - 1
                    break
                else:
                    if data[middle + 1] == k:
                        left = middle
                        last = middle + 1
                    else:
                        last = middle
                        break
            elif data[middle] < k:
                left = middle
            else:
                right = middle
        print(left, right)
        print(first, last)
        if exist == 1:
            return last - first + 1
        else:
            return 0
```

## 题目二：0~n-1中缺失的数字

>一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0~n-1内。在范围0~n-1内的n个数字中有且仅有一个数字不在
数组中，请找出这个数字。

解法分析：直观解法是n(n-1)/2减去数组的和，得到的就是那个缺失的数字。但是时间复杂度是O(n)。分析规律，有且仅有一个缺失的数字，
因此在缺失数字之前，每个索引对应的值等于索引，在缺失数字之后（包含缺失数字）所有索引对应的数字都不等于索引。只需要二分查找第一个
索引和值不对应的索引即可。如果中间数字和索引相等，则把left移到middle+1的位置；如果相等且前一个数字和索引也相等，则把right移到
middle-1的位置；如果相等且前一个数字和索引不相等，那么当前索引是第一个开始不相等的位置，索引值就是缺失值。进入while循环的条件是
left <= right。

## 题目三：数组中数值和下标相等的元素

>假设一个单调递增的数组里的每个元素都是整数并且是唯一的。请编程实现一个函数，找出数组中任意一个数值等于其下标的元素。例如，在[-3, 
-1, 1, 3, 5]中，数字3和它的下标相等。

解法分析：最直观的解法还是遍历一遍数组，然后对每个数字判断是否和下标相同。时间复杂度O(n)。由于是单调递增序列，因此可以使用二分查找。
分析数组的规律知，如果一个数字的值大于它的下标，那么它后面的所有值也都大于下标。因此如果下标middle的数字大于它的下标，那么在其左边
寻找即可；否则在其右边寻找。

```python
def get_number_same_as_index(numbers):
    if len(numbers) == 0:
        return -1
    left = 0
    right = len(numbers) - 1
    while left <= right:
        middle = (left + right) // 2
        #print(left, middle, right)
        if numbers[middle] == middle:
           return middle
        if numbers[middle] > middle:
            right = middle - 1
        else:
            left = middle + 1
    return -1

print(get_number_same_as_index([-3,-1,1,3,5])) # 3
print(get_number_same_as_index([0,3,4,5,7])) # 0
print(get_number_same_as_index([-1,0,1,2,4])) # 4
```

# 面试题54：二叉搜索树的第K小节点

>给定一棵二叉搜索树，请找出其中第K小的节点。

解法分析：实际考察的是对二叉树中序遍历的理解。**如果中序遍历一棵二叉搜索树，那么遍历序列就是递增排序的**。注意返回的是节点，
而不是节点的值。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.arr = []
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        if k <= 0 or not pRoot:
            return None
        self.middle_traverse(pRoot)
        if len(self.arr) < k:
            return None
        else:
            return self.arr[k-1]
        
    def middle_traverse(self, pRoot):
        if pRoot is None:
            return None
        self.middle_traverse(pRoot.left)
        self.arr.append(pRoot)
        self.middle_traverse(pRoot.right)
```

# 面试题55：二叉树的深度

## 题目一：二叉树的深度

>输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

解法1：找到树的所有路径，然后返回最大的长度值。代码量比较大，不够简洁。

解法2：递归的做法。如果一棵树根节点为None，那么深度为0；否则，这棵树的深度等于max(左子树深度,右子树深度) + 1.

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        if left > right:
            return left + 1
        else:
            return right + 1
```

## 题目二：平衡二叉树

>输入一棵二叉树，判断该二叉树是否是平衡二叉树。

解法1：如果根节点不为None，那么判断其左右子树深度的差的绝对值是不是小于等于1，如果不是，则返回False；如果是，则返回其左子树和
右子树的判断结果相与之后的值。代码简洁但是效率不高，因为一个节点会被重复遍历多次。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        """
        它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，
        并且左右两个子树都是一棵平衡二叉树
        """
        if not pRoot:
            return True
        if abs(self.TreeDepth(pRoot.left) - self.TreeDepth(pRoot.right)) > 1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
    
    def TreeDepth(self, root):
        if not root:
            return 0
        left = self.TreeDepth(root.left)
        right = self.TreeDepth(root.right)
        
        return max(left+1, right+1)
```

解法2：对上面的算法进行优化。在计算深度时判断是否为平衡二叉树，如果不是则直接返回-1；如果是则继续。在主函数里调用获取深度的函数，
如果得到的不是-1，那么是平衡二叉树，否则不是平衡二叉树。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if pRoot is None:
            return True
        return self.get_depth(pRoot) != -1
    
    def get_depth(self, pRoot):
        if pRoot is None:
            return 0
        left = self.get_depth(pRoot.left)
        if left == -1:
            return -1
        right = self.get_depth(pRoot.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        else:
            return max(left+1, right+1)
```

# 面试题56：和为s的数字

## 题目一：和为s的两个数字

>输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

解法：哈希表

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if len(array) < 2:
            return []
        res = []
        d = set()
        for val in array:
            if tsum - val in d:
                res.append([tsum - val, val])
            else:
                d.add(val)
        if len(res) == 0:
            return []
        min_product = res[0][0] * res[0][1]
        result = res[0]
        for l in res:
            if l[0] * l[1] < min_product:
                min_product = l[0] * l[1]
                result = l
        return result[0], result[1]
```

## 题目二：和为s的连续正数序列

>小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的
正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的
找出所有和为S的连续正数序列? Good Luck!  输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序

```python
class Solution:
    def FindContinuousSequence(self, tsum):
        if tsum < 3:
            return []
        small = 1
        big = 2
        middle = (tsum + 1)>>1
        curSum = small + big
        output = []
        while small < middle:
            if curSum == tsum:
                output.append(list(range(small, big+1))) # 注意python2和3range的区别。python3外面需要加list()。
                big += 1
                curSum += big
            elif curSum > tsum:
                curSum -= small
                small += 1
            else:
                big += 1
                curSum += big
        return output
print(Solution().FindContinuousSequence(9))
```

# 面试题58：翻转字符串

## 题目一：翻转单词顺序

>输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student."，
则输出"student. a am I"。

解法：

```python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        if not s:
            return s
        l = s.split(" ")
        l.reverse()
        res = ""
        for val in l[:-1]:
            res += val
            res += " "
        res += l[-1]
        return res
```

## 题目二：左旋转字符串

>汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，
请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

```python
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if n == 0:
            return s
        if not s:
            return s
        return s[n:] + s[:n]
```

# 面试题59：队列的最大值

## 题目一：滑动窗口的最大值

>给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，
那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}，
 {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

解法：双端队列。用一个叫window的队列存储当前滑动窗口中可能是最大值的索引。首先用一个while循环把窗口范围之前的值去掉，然后删除掉window中
对应值小于当前值的那些值（因为先要要append进来的值比他们来得晚而且还比他们大，因此那些值永远无出头之日，删掉即可）。然后把当前值添加
到window中。每次把window中的第0个元素在num中的值添加到res中。最后返回res即可。

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if len(num) < size or size < 1:
            return []
        res = []
        window = []  # index of val in window
        for i, val in enumerate(num):
            while len(window) >= 1 and i >= size and window[0] <= i - size:  # 满足则从队列前面删除
                window.pop(0)
            while len(window) >= 1 and val >= num[window[-1]]:  # 满足则从队列后面删除
                window.pop(-1)
            window.append(i)
            if i >= size - 1:
                res.append(num[window[0]])
        return res
```

## 题目二：队列的最大值

>定义一个队列并实现函数max得到队列里的最大值，要求函数max、push_back和push_font的时间复杂度都是O(1)


```python

```

# 面试题60：n个骰子的点数

>把n个骰子仍在地上，所有骰子朝上一面的点数之和为s。输出n，打印出s的所有可能的值出现的概率。

```python
class Solution:
    def dices_sum(self, n):
        # Write your code here
        if n == 0: return None
        result = [
            [1, 1, 1, 1, 1, 1],
        ]
        # if n == 1: return result[0]
        # 计算n个骰子出现的各个次数和
        for i in range(1, n):
            x = 5 * (i + 1) + 1
            result.append([0 for _ in range(x)])

            for j in range(x):
                if j < 6:
                    result[i][j] = (sum(result[i - 1][0:j + 1]))
                elif 6 <= j <= 3 * i + 2:
                    result[i][j] = (sum(result[i - 1][j - 5:j + 1]))
                else:
                    break
            left = 0
            right = len(result[i]) - 1
            while left <= right:
                result[i][right] = result[i][left]
                left += 1
                right -= 1

        res = result[-1]
        all = float(sum(res))
        other = []
        # 第i个元素代表骰子总和为n+i
        for i, item in enumerate(res):
            pro = item / all
            other.append([n + i, pro])
        return other
```

# 面试题61：扑克牌中的顺子

>题目：从扑克牌中随机抽五张牌，判断是不是一个顺子，即这五张牌是不是连续的。2~10为数字本身，A为1，J为11，Q为12，K为13，而大小王可以看成
任意数字。

解法：（1）把数组排序；（2）统计0的个数和相邻数字之间的空缺总数（除了0外相邻数字之间如果相等就返回False）；（3）如果0的个数不少于
空缺总数，则这个数组就是连续的，反正不连续。

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if not numbers:
            return False
        numbers.sort()
        num_0 = 0
        num_vacancy = 0
        for i, val in enumerate(numbers):
            if val == 0:
                num_0 += 1
            elif i != 4 and numbers[i+1] != numbers[i]:
                num_vacancy += (numbers[i+1] - numbers[i] - 1)
            elif i != 4 and numbers[i+1] == numbers[i]:
                return False
        if num_vacancy <= num_0:
            return True
        else:
            return False
```

# 面试题62：圆圈中最后剩下的数字

>题目：0,1,...,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

经典解法：环形链表模拟圆圈。创建一个共有n个节点的环形链表，然后每次在这个链表中删除第m个节点。用list模拟环形链表的方法，当扫描到list
末尾的时候，把它移到头部即可。

最佳解法：要得到n个数字序列中最后剩下的数字，只需要得到n-1个数字的序列中最后剩下的数字，并以此类推。当n=1时，也就是序列只有一个0时，结果
就是0.f(n, m) = 0 如果n=1；f(n, m) = (f(n-1, m) + m) % n 如果n>1.


```python
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n < 1 or m < 1:
            return -1
        last = 0
        for i in range(2, n+1):
            last = (last + m) % i
        return last
```

# 面试题63：股票的最大利润

>假设把某股票的价格按照时间先后顺序存储在数组中，请问**买卖该股票一次**可获得的最大利润是多少？

解法：以此扫描数组中的数字，当扫描第i个数字时，只要我们能记住之前i-1个数字中的最小值，然后算出当前价位卖出的最大利润。最后取最大的即可。

```python
def stock(nums):
    if not nums or len(nums) == 1:
        return 0
    min_val = nums[0]
    max_diff = nums[1] - min_val
    for i in range(2, len(nums)):
        if nums[i-1] < min_val:
            min_val = nums[i-1]
        if nums[i] - min_val > max_diff:
            max_diff = nums[i] - min_val
    return max_diff

print(stock([9, 11, 8, 5, 7, 12, 16, 14])) # 11
```

# 面试题64：求1+2+3+...+n

>要求不能使用乘除法, for, while, if else, switch, case等关键字以及条件判断语句(A?B:C)

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.sum = 0
    def Sum_Solution(self, n):
        # write code here
        self.getsum(n)
        return self.sum
    
    def getsum(self, n):
        self.sum += n
        n -= 1
        return n > 0 and self.getsum(n)
```

# 面试题65：不用加减乘除做加法

>题目写一个函数，求两个整数之和，要求在函数体内不得使用"+", "-", "x", "/"四则运算符号。

解法：首先对每一位进行异或，然后比较对应较小的一位是不是都是1，如果是则需要进位，当前位需要加1（求与并左移一位，得到的数和异或的数进行异或），
直到没有进位为止。

```python
class Solution:
    def Add(self, num1, num2):
        # write code here
        sum = 0
        carry = 0
        while num2:
            sum = num1 ^ num2
            carry = (num1 & num2) << 1
            num1, num2 = sum, carry
            print(num1, num2)
        return num1
print(Solution().Add(5, 17))
print(Solution().Add(5, 20))
```

# 面试题67：把字符串转成整数

```python
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        if len(s) == 0:
            return 0
        dic = {str(i): i for i in range(10)}
        if s[0] != '+' and s[0] != '-' and s[0] not in dic:
            return 0
        for val in s[1:]:
            if val not in dic:
                return 0
        if s[0] == "+":
            return self.str_to_int(s[1:])
        elif s[0] == "-":
            return -1 * self.str_to_int(s[1:])
        else:
            return self.str_to_int(s)
    
    def str_to_int(self, s):
        res = 0
        for val in s:
            res = res * 10 + int(val)
        return res
```






















