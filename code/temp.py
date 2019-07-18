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
        数学上可以证明，当n>=5时，尽可能多的剪长度为3的绳子，且最后一段绳子如果是4的话，把它剪成2+2的两段
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
