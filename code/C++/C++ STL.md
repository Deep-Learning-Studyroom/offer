# vector
[C++中vector使用详细说明 (转)](https://www.cnblogs.com/aminxu/p/4686332.html)

## 一维向量
**向量的声明及初始化**

```c++
vector<int> a ;                                //声明一个int型向量a
vector<int> a(10) ;                            //声明一个初始大小为10的向量
vector<int> a(10, 1) ;                         //声明一个初始大小为10且初始值都为1的向量
vector<int> b(a) ;                             //声明并用向量a初始化向量b
vector<int> b(a.begin(), a.begin()+3) ;        //将a向量中从第0个到第2个(共3个)作为向量b的初始值

int n[] = {1, 2, 3, 4, 5} ;
vector<int> a(n, n+5) ;              //将数组n的前5个元素作为向量a的初值
vector<int> a(&n[1], &n[4]) ;        //将n[1] - n[4]范围内的元素作为向量a的初值
```
**向量的基本操作**

```c++
a.size()                 //获取向量中的元素个数
a.empty()                //判断向量是否为空
a.clear()                //清空向量中的元素
a = b ;                  //将b向量复制到a向量中
a == b ;                 //a向量与b向量比较, 相等则返回1

a.insert(a.begin(), 1000);            //将1000插入到向量a的起始位置前
a.insert(a.begin(), 3, 1000) ;        //将1000分别插入到向量元素位置的0-2处(共3个元素)

vector<int> a(5, 1) ;
vector<int> b(10) ;
b.insert(b.begin(), a.begin(), a.end()) ;        //将a.begin(), a.end()之间的全部元素插入到b.begin()前

b.erase(b.begin()) ;                     //将起始位置的元素删除
b.erase(b.begin(), b.begin()+3) ;        //将(b.begin(), b.begin()+3)之间的元素删除

b.swap(a) ;            //a向量与b向量进行交换
swap(a,b)             // 将a和b元素互换。同上操作。

c.pop_back()       // 删除最后一个数据。
c.push_back(elem)  // 在尾部加入一个数据。



```

## 二维向量


```c++
vector< vector<int> > b(10, vector<int>(5));        //创建一个10*5的int型二维向量
```

```c++
b.size()            //获取行向量的大小
b[0].size()         //获取b[0]向量的大小
```

## code

```c++
#include <iostream>
#include<vector>
#include<queue>
#include<deque>
#include<map>
using namespace std;


int main()
{
	cout << "1d vector\n";
	int num[] = { 0,1,2,3,4,5,6,7,8,9 };
	vector<float> a(num, num+10);
	
	a.push_back(11);
	a.pop_back();
	a[9] = 9.1;
	for (int i=0; i<a.size();i++)
	    cout << a[i] << " ";
	cout << endl;

	cout << "2d vector\n";
	vector<vector<float>>  b(10, a);

	cout << b.size() << " " << b[0].size() << endl;

	return 0;
}
```

