/*
二分查找及其四种变种：
1. 没有重复元素的排序数组中查找某个数；
2. 查找第一个等于给定值的元素；
3. 查找最后一个等于给定值的元素；
4. 查找第一个大于等于给定值的元素；
5. 查找第一个小于等于给定值的元素。
*/

#include<iostream>
using namespace std;

//mid的计算，相加除以二有可能超出int范围，右移一位等于除以2（注意运算符的级别）
int bsearch(int* nums, int n, int value)
{
	int low = 0;
	int high = n - 1;

	while (low <= high)
	{
		int mid = ((high - low) >> 1) + low;
		cout << mid << " ";
		if (nums[mid] < value)
		{
			low = mid + 1;
		}
		else if (nums[mid] > value)
		{
			high = mid - 1;
		}
		else
		{
			cout << endl;
			return mid;
		}
	}
	cout << endl;
	return -1;
}

//查找第一个等于给定值的元素，返回索引；
int bsearch_1(int* nums, int n, int value)
{
	int low = 0;
	int high = n - 1;

	while (low <= high)
	{
		int mid = ((high - low) >> 1) + low;
		cout << mid << " ";
		if (nums[mid] < value)
		{
			low = mid + 1;
		}
		else if (nums[mid] > value)
		{
			high = mid - 1;
		}
		else
		{
			if (mid == 0 || nums[mid - 1] != value)
			{
				cout << endl;
				return mid;
			}
			else
			{
				high = mid - 1;
			}
			
		}
	}
	cout << endl;
	return -1;
}

//查找最后一个等于给定值的元素，返回索引；
int bsearch_2(int* nums, int n, int value)
{
	int low = 0;
	int high = n - 1;
	
	while (low <= high)
	{
		int mid = ((high - low) >> 1) + low;
		cout << mid << " ";
		if (nums[mid] < value)
		{
			low = mid + 1;
		}
		else if (nums[mid] > value)
		{
			high = mid - 1;
		}
		else
		{
			if (mid == n-1 || nums[mid + 1] != value)
			{
				cout << endl;
				return mid;
			}
			else
			{
				low = mid + 1;
			}

		}
	}
	cout << endl;
	return -1;
}

//查找第一个大于等于给定值的元素，返回索引
int bsearch_3(int* nums, int n, int value)
{
	int low = 0;
	int high = n - 1;
	while (low <= high)
	{
		int mid = ((high - low) >> 2) + low;
		cout << mid << " ";
		if (nums[mid] >= value)
		{
			if (mid == 0 || nums[mid - 1] < value)
			{
				cout << endl;
				return mid;
			}	
			else
			{
				high = mid - 1;
			}
		}
		else
		{
			low = mid + 1;
		}
	}
	cout << endl;
	return -1;
}

//查找最后一个小于等于给定值的元素，返回索引
int bsearch_4(int* nums, int n, int value)
{
	int low = 0;
	int high = n - 1;
	while (low <= high)
	{
		int mid = ((high - low) >> 2) + low;
		cout << mid << " ";
		if (nums[mid] <= value)
		{
			if (mid == n-1 || nums[mid + 1] > value)
			{
				cout << endl;
				return mid;
			}
			else
			{
				low = mid + 1;
			}
		}
		else
		{
			high = mid - 1;
		}
	}
	cout << endl;
	return -1;
}

int main()
{
	//原始二分法
	int nums[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	cout << bsearch(nums, 10, 8) << endl;

	//查找第一个等于给定值的元素，返回索引；
	int nums_1[10] = { 0, 1, 2, 3, 4, 5, 6, 8, 8, 9 };
	cout << bsearch_1(nums_1, 10, 8) << endl;

	//查找最后一个等于给定值的元素，返回索引；
	cout << bsearch_2(nums_1, 10, 8) << endl;

	//查找第一个大于等于给定值的元素，返回索引
	cout << bsearch_3(nums_1, 10, 7) << endl;

	//查找最后一个小于等于给定值的元素，返回索引
	cout << bsearch_4(nums_1, 10, 8) << endl;

	return 0;
}