# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        for row in range(rows):
            for col in range(cols):
                if matrix[row * cols + col] == path[0]:
                    if self.find(list(matrix), rows, cols, path[1:], row, col):
                        return True
        return False

    def find(self, matrix, rows, cols, path, row, col):
        if not path:
            return True
        print(row, col, matrix[row * cols + col])
        matrix[row * cols + col] = '0'
        for i in range(rows):
            for j in range(cols):
                print(matrix[i * cols + j], end=" ")
            print()
        print()
        if col + 1 <= cols - 1 and matrix[row * cols + (col + 1)] == path[0]:
            return self.find(matrix, rows, cols, path[1:], row, col + 1)
        elif col - 1 >= 0 and matrix[row * cols + (col - 1)] == path[0]:
            return self.find(matrix, rows, cols, path[1:], row, col - 1)
        elif row + 1 <= rows - 1 and matrix[(row + 1) * cols + col] == path[0]:
            return self.find(matrix, rows, cols, path[1:], row + 1, col)
        elif row - 1 >= 0 and matrix[(row - 1) * cols + col] == path[0]:
            return self.find(matrix, rows, cols, path[1:], row - 1, col)
        else:
            return False

solution = Solution()

#print(solution.hasPath("ABCESFCSADEE",3,4,"SEC"))
#print(solution.hasPath("ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS",5,8,"SGGFIECVAASABCEHJIGQEM"))
#print(solution.hasPath("ABTGCFCSJDEH",3,4,"BFCE"))
print(solution.hasPath("ABTGCFCSJDEH",3,4,"BFCJ"))