def big_number_sum(s1, s2):
    """首先写出两个整数相加和相减的函数，然后根据两个数的正负号分四种类别计算"""
    if s1[0] != '-' and s2[0] != '-':
        big_positive_number_sum(s1, s2)
    elif s1[0] == '-' and s2[0] != '-':
        s1 = s1[1:]
        big_positive_number_minus(s2, s1)
    elif s1[0] != '-' and s2[0] == '-':
        s2 = s2[1:]
        big_positive_number_minus(s1, s2)
    else:
        s1 = s1[1:]
        s2 = s2[1:]
        print('-', end="")
        big_positive_number_sum(s1, s2)

def big_positive_number_sum(s1, s2):
    """两个大正数相加，首先对齐(补0)，然后按加法计算规则计算"""
    L1=[0]
    L2=[0]
    for i in range(0,len(s1)):
        L1.append(int(s1[i]))
    for i in range(0,len(s2)):
        L2.append(int(s2[i]))

    if(len(s1)>len(s2)):
        for i in range(len(s1)-len(s2)):
            L2.reverse()
            L2.append(0)
            L2.reverse()
    elif(len(s1)<=len(s2)):
        for i in range(len(s2)-len(s1)):
            L1.reverse()
            L1.append(0)
            L1.reverse()

    for i in range(len(L1)):
            L1[i]=L1[i]+L2[i]
    A=B=len(L1)-1

    while A>0:
        if((L1[A])/10)>=1:
            L1[A]=L1[A]%10
            L1[A-1]=L1[A-1]+1
        A-=1
    if L1[0]==0:
        for i in range(1,B+1):
            print(L1[i],end='')
    elif L1[0]!=0:
        for i in range(B+1):
            print(L1[i],end='')
    print()

def big_positive_number_minus(s1, s2):
    """减法：注意一点，如果s1减去s2得到的第一位的数字是-1，那么说明s2更大，先打印一个负号，然后调用big_positive_number_minus(s2, s1)"""
    L1=[0]
    L2=[0]
    for i in range(0,len(s1)):
        L1.append(int(s1[i]))
    for i in range(0,len(s2)):
        L2.append(int(s2[i]))

    if(len(s1)>len(s2)):
        for i in range(len(s1)-len(s2)):
            L2.reverse()
            L2.append(0)
            L2.reverse()
    elif(len(s1)<=len(s2)):
        for i in range(len(s2)-len(s1)):
            L1.reverse()
            L1.append(0)
            L1.reverse()

    for i in range(len(L1)):
            L1[i]=L1[i] - L2[i]
    A=B=len(L1)-1

    while A>0:
        if L1[A] < 0:
            L1[A]=L1[A] + 10
            L1[A-1]=L1[A-1] - 1
        A-=1
    if L1[0]==0:
        for i in range(1, B+1):
            print(L1[i],end='')
            if i == B:
                print()
    elif L1[0]!=0:
        print('-', end="")
        big_positive_number_minus(s2, s1)

big_number_sum('1323479817', '1372987318423498414')
big_number_sum('1323479817', '-1372987318423498414')
big_number_sum('-1323479817', '1372987318423498414')
big_number_sum('-1323479817', '-1372987318423498414')
print()
print(1323479817 + 1372987318423498414)
print(1323479817 - 1372987318423498414)
print(-1323479817 + 1372987318423498414)
print(-1323479817 - 1372987318423498414)