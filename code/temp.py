# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0:
            return 0
        if exponent == 1:
            return base

        if exponent % 2 == 0:
            return self.Power(base, exponent / 2) * self.Power(base, exponent / 2)
        if exponent % 2 == 1:
            return self.Power(base, (exponent - 1) / 2) * self.Power(base, (exponent - 1) / 2) * base