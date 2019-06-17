def fibo(n):
    first_number = 0
    second_number = 1

    if n == 0:
        return 0
    elif n == 1:
        return 1
    for _ in range(n):
        first_number, second_number = second_number, first_number+second_number
    return first_number

if __name__ == '__main__':
    print(fibo(5))