def mySqrt(x):
        
    if x < 0:
        print("error")
        return 
    if x == 0 or x == 1:
        return x
    
    l = 0
    r = x
    
    
    while abs(l-r) > 1e-2:
        mid = (l + r) / 2
        y = mid * mid
        if y < x:
            l = mid
        elif y > x:
            r = mid
        else:
            return int(mid)
    if (int(mid) + 1) ** 2 == x:
        print(l,mid, r)
        return int(mid) + 1
    elif int(mid) ** 2 == x:
        return int(mid)
    elif int(mid) ** 2 > x:
        return int(mid) - 1
    else:
        print(l,mid, r)
        return int(mid)
    

if __name__ == '__main__':
    print(mySqrt(1464898982))