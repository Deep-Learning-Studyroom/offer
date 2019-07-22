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