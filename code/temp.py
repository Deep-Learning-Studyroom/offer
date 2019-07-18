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