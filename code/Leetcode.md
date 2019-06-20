

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

# [leetcode-20 valid-parentheses(括号匹配问题)](https://leetcode.com/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        if not str:
            return true
        stack = []
        d = {'(':0, '[':0, '{':0, ')':1, ']':1, '}':1}
        d2 = {')':'(', ']':'[', '}':'{'}
        for i in s:
            if i not in d:
                raise ValueError
            if d[i] == 0:
                stack.append(i)
            else:
                if stack and d2[i] == stack[-1]:
                    stack.pop()
                else:
                    return False
        if stack:
            return False
        else:
            return True
```
# [leetcode-206 reverse-linked-list](https://leetcode.com/problems/reverse-linked-list/)

```python
#Definition for singly-linked list.
#class ListNode:
#    def __init__(self, x):
#        self.val = x
#        self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            cur.next, pre, cur = pre, cur, cur.next
        return pre
```

**注意：链表的问题想法都很简单，主要考察代码实现能力**  

# [swap-nodes-in-pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)  

```python3


```

**有难度，很容易写错**  

# [leetcode-141 linked-list-cycle](https://leetcode.com/problems/linked-list-cycle/)

**面试高频题** 给定一个链表，判断是否有环  

解法1：硬做。不断看next，如果发现有None就返回False(没有)，如果在一定时间内都找不到None则返回True。原因，有环的链表的任意一个元素next指针不为None。  

解法2：set，判断是否重复。 O(n) 

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        nodes = set()
        
        current_node = head
        while current_node:
            if current_node in nodes:
                return True
            else:
                nodes.add(current_node)
            current_node = current_node.next
            
        return False
```
解法3：快慢指针。快指针每次走两步，慢指针每次走一步。最后判断快与慢相遇。O(n)，但是空间复杂度比解法2低。

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        fast = slow = head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False
```


# [leetcode242 valid-anagram](https://leetcode.com/problems/valid-anagram/)

解法1：使用快速排序分别排序两个字符串，然后比较两者是否相同。时间复杂度O(nlog(n))。

解法2：使用map计数。即使分别用一个字典记录对于每一个字符串其中每个字符出现的次数。然后比较是否相同。由于字典的插入、删除和索引都是O(1)的，因此该算法时间复杂度是O(n)。

解法3:使用map计数。但是不使用字典结构，而是使用列表构建哈希表，哈希函数就是每个元素的ASCII码减去a的ASCII码。这种方法比解法2还会稍微快一点。时间复杂度也是O(n)。

解法2代码  
```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        dict1, dict2 = {}, {}
        for i in s:
            dict1[i] = dict1.get(i, 0) + 1
        for j in t:
            dict2[j] = dict2.get(j, 0) + 1
        return dict1==dict2
```
**这里需要注意一下，字典的get方法比直接dict[i]来获取value更好，因为前者在没有这个键的时候不会报错，会返回设置的default值**

解法3代码  
```python
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        dic1, dic2 = [0] * 26, [0] *26
        for i in s:
            dic1[ord(i) - ord('a')] += 1
        for j in t:
            dic2[ord(j) - ord('a')] += 1
        return dic1 == dic2
```

# [leetcode-98 validate-binary-search-tree](https://leetcode.com/problems/validate-binary-search-tree/)

解法1：中序遍历。看遍历出来的序列是不是**升序**的。时间复杂度是O(n)  

解法2：recursive。validate(···, min, max) 这个函数要向外面传两个值，min和max。validate(node.left) --> max, validate(node.right) --> min ，max < root, min > root即可。时间复杂度是O(n)

解法1代码  
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        inorder = self.inorder(root)
        
        return inorder == list(sorted(set(inorder)))
    def inorder(self, root):
        if root is None:
            return []
        return self.inorder(root.left) + [root.val] + self.inorder(root.right)
```
**注意：由于之前不熟悉二叉树，函数里面什么时候用root，什么时候用self容易搞混**

Runtime: 44 ms, faster than 29.23% of Python online submissions for Validate Binary Search Tree.  
Memory Usage: 17.5 MB, less than 5.03% of Python online submissions for Validate Binary Search Tree.  
**从上面可以看出空间复杂度非常大，修改方法就是每次遍历不需要把所有的都存下来，只需要判断当前节点是否比前继节点满足递增的关系，如果不是就返回False，如果满足就继续判断**

解法1的改进代码  
```python 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.prev = None
        return self.helper(root)
    
    def helper(self, root):
        if root is None:
            return True
        if not self.helper(root.left):
            return False
        if self.prev and self.prev.val >= root.val:
            return False
        self.prev = root
        return self.helper(root.right)
```
Runtime: 28 ms, faster than 97.82% of Python online submissions for Validate Binary Search Tree.  
Memory Usage: 16.8 MB, less than 20.61% of Python online submissions for Validate Binary Search Tree.  

解法2代码：  
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root, lo=float('-inf'), hi=float('inf')):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        def BST(n, min_val, max_val):
            if not n:
                return True
            if not min_val < n.val < max_val:
                return False
            return BST(n.left, min_val, n.val) and BST(n.right, n.val, max_val)         
        return BST(root,-float("inf"), float("inf"))
```

# [leetcode-236 lowest-common-ancestor-of-a-binary-tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)  

对于一个普通二叉树（不一定是二叉搜索树），找p和q的共同最近祖先  

解法1：分别从p和q往回走，记录下路径，然后找两个路径最开始相遇的点。但是由于二叉树没有parent指针，所以这个方法没法实现。但是可以从root节点遍历找p, q，然后比较路径上最后相同的点。时间复杂度是O(n)，但是是3n，而且实现起来不简单。

解法2：recursion。 设计一个函数findPorQ(root, p, q)， 如果root等于p或者q就返回root，否则就分别在root.left和root.right两个子树调用findPorQ。时间复杂度O(n)。

解法2的代码  
```python 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root == None:
            return None
        if root == p or root == q:
            return root
        tree_node_left = self.lowestCommonAncestor(root.left, p, q)
        tree_node_right = self.lowestCommonAncestor(root.right, p, q)
        if tree_node_left == None:
            return tree_node_right
        elif tree_node_right == None:
            return tree_node_left
        else:
            return root
```
**这个算法非常好，多想一想**

# [leetcode-235 lowest-common-ancestor-of-a-binary-search-tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

解法：这是236的简化版本。用递归的方法做，如果p/q都比root小，所以它们应该在左子树里面；如果p/q都比root大，那么它们应该在右子树里面，如果一个比root小，一个比root大，那么返回root。

代码
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if p.val < root.val > p.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
```
**注意：这个代码运行出错了，感觉很奇怪！**   

非递归的做法

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        while root:
            if p.val < root.val > q.val:
                root = root.left
            elif p.val > root.val < q.val:
                root = root.right
            else:
                return root
 






