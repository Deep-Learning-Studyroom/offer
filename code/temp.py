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