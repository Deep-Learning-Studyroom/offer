def quick_sort(nums):
    if len(nums) == 0 or len(nums) == 1:
        return nums
    
    mid_value = nums[0]
    below = [x for x in nums[1:] if x <= mid_value]
    above = [x for x in nums[1:] if x > mid_value]

    return quick_sort(below) + [mid_value] + quick_sort(above)

if __name__ == '__main__':
    print(quick_sort([1,4,2,6,3,10,2]))