def fibo(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    return fibo(n-1) + fibo(n-2)
    

if __name__ == '__main__':
    print(fibo(5))