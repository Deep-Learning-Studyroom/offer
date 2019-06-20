from heapq import *
def kth_largest(num, k):
    if len(num) < k:
        raise ValueError
    heap = num[:k]
    heapify(heap) # 默认是小顶堆
    for i in num[k:]:
        if i <= heap[-1]:
            pass
        else:
            heappop(heap)
            heappush(heap, i)
        #heapify(heap)
    return heap






if __name__ == '__main__':
    print(kth_largest([4,5,6,7,1,2,3], 2))