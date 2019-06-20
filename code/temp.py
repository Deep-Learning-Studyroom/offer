'''
def partition(num, low, high):
    pivot = num[low] 
    while low < high:
        while low < high and pivot < num[high]:
            high -= 1
        while low < high and num[low] < pivot:
            low += 1
        num[low], num[high] = num[high], num[low]
        num[high] = pivot
    return high, num

def findkth(num, low, high, k):
    index = (partition(num, low, high))[0]
    #print(partition(num, low, high)[1])
    if index == k:
        return num[index]
    elif index < k:
        return findkth(num, index+1, high, k)
    else:
        return findkth(num, low, index-1, k)
'''
def partition(num, low, high):
    pivot = num[low]
    while (low < high):
        while (low < high and num[high] > pivot):
            high -= 1
        while (low < high and num[low] < pivot):
            low += 1
        num[low],num[high] = num[high],num[low]
    num[high] = pivot
    return high,num

def findkth(num,low,high,k):   #找到数组里第k个数
        index=(partition(num,low,high))[0]
        print((partition(num,low,high))[1])
        if index==k:return num[index]
        if index<k:
            return findkth(num,index+1,high,k)
        else:
            return findkth(num,low,index-1,k)    

if __name__ == '__main__':
    print(findkth([6,1,3,9,2],0,len([6,1,3,9,2])-1,4))