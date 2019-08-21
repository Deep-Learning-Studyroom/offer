#include<iostream>
using namespace std;

// 直接插入排序
void insert_sort(int a[], int n)
{
	int i, j, temp;
	for (i = 1; i < n; i++)  //需要选择n-1次
	{
		// 暂存下标为1的数，下标从1开始，因为开始时下标为0的数可以认为是排好序的
		temp = a[i];
		for (j = i - 1; j >= 0 && temp < a[j]; j--)
		{
			//如果满足条件就往后挪，最坏的情况就是temp比a[0]小
			a[j + 1] = a[j];
		}
		a[j + 1] = temp; //找到下标为i的数放置位置
	}
}

// 希尔排序
void shell_sort(int a[], int len)
{
	int h, i, j, temp;
	for (h = len / 2; h > 0; h = h / 2)  //控制增量
	{
		for (i = h; i < len; i++)    //这个for循环就是直接插入排序
		{
			temp = a[i];
			for (j = i - h; (j >= 0 && temp < a[j]); j -= h)
			{
				a[j + h] = a[j];
			}
			a[j + h] = temp;
		}
	}
}

// 冒泡排序
void bubble_sort_1(int a[], int len)
{
	int i = 0;
	int j = 0;
	int temp = 0;
	for (i = 0; i < len - 1; i++)             //进行n-1趟扫描
	{
		for (j = len - 2; j >= i; j--)        //从后往前交换，这样最小值冒泡到开头部分
		{
			if (a[j+1] < a[j])              
			{
				temp = a[j];
				a[j] = a[j + 1];
				a[j + 1] = temp;
			}
		}
	}
}

void bubble_sort_2(int a[], int len)
{
	int i = 0;
	int j = 0;
	int temp = 0;
	int exchange = 0;                         //用于记录每次扫描时是否发生交换 

	for (i = 0; i < len - 1; i++)             //进行n-1趟扫描
	{
		exchange = 0;
		for (j = len - 2; j >= i; j--)        //从后往前交换，这样最小值冒泡到开头部分
		{
			if (a[j + 1] < a[j])
			{
				temp = a[j];
				a[j] = a[j + 1];
				a[j + 1] = temp;
				exchange = 1;
			}
		}
		if (exchange != 1)
		{
			return;  // 此次扫描没有发生交换，说明已经排好序的
		}
	}
}

//快排
int partition(int a[], int low, int high)
{
	int pivot = a[low];
	int temp;
	while (low < high)
	{
		while (low < high && a[high] > pivot) high--;
		if (low < high) a[low++] = a[high];
		while (low < high && a[low] < pivot) low++;
		if (low < high) a[high--] = a[low];
	}
	a[low] = pivot;
	return low;
}
void quick_sort(int a[], int low, int high)
{
	if (low < high)
	{
		int location = partition(a, low, high);
		quick_sort(a, low, location - 1);
		quick_sort(a, location + 1, high);
	}
	cout << "1";
}

void quick_sort_2(int a[], int low, int high)
{
	int i, j, pivot, temp;
	if (low < high)
	{
		pivot = a[low];
		i = low;
		j = high;
		while (i < j)
		{
			while (i < j && a[j] >= pivot)
				j--;

			if (i < j) a[i++] = a[j];  //将比pivot小的元素移动到低端
			while (i < j && a[i] <= pivot)
				i++;
			if (i < j) a[j--] = a[i];  //将比pivot大的元素移动到高端
		}

		a[i] = pivot;                  //pivot移到最终位置
		quick_sort_2(a, low, i - 1);
		quick_sort_2(a, i + 1, high);
	}
}

void print_array(int a[], int len)
{
	for (int i = 0; i < len; i++)
	{
		cout << a[i] << " ";
	}
	cout << endl;
}



int main()
{
	int a[] = {3,1,5,2,8,2,0};
	cout << "a[]中的a:" << a << endl;
	cout << "before sort: \n";
	print_array(a, 7);
	/*
	insert_sort(a, 7);
	cout << "after insert sort: \n";
	print_array(a, 7);

	shell_sort(a, 7);
	cout << "after shell sort: \n";
	print_array(a, 7);
	
	bubble_sort_1(a, 7);
	cout << "after bubble sort 1: \n";
	print_array(a, 7);
	//待优化点：假设第i次扫描前，数组已经排好序了，但是它还会进行下一次的扫描
	//这是没有必要的，优化如下
	
	bubble_sort_2(a, 7);
	cout << "after bubble sort 2: \n";
	print_array(a, 7);
	*/
	quick_sort(a, 0, 6);
	//quick_sort_2(a, 0, 6);
	cout << "after quick sort: \n";
	print_array(a, 7);

	return 0;
}