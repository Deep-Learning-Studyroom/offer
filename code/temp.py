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

#my_print(3)
my_print2(3)