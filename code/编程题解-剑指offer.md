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

