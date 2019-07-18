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