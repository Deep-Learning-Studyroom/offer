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

# queue

## 基本操作

queue入队，如例：q.push(x); 将x 接到队列的末端。

queue出队，如例：q.pop(); 弹出队列的第一个元素，注意，并不会返回被弹出元素的值。

访问queue队首元素，如例：q.front()，即最早被压入队列的元素。

访问queue队尾元素，如例：q.back()，即最后被压入队列的元素。

判断queue队列空，如例：q.empty()，当队列空时，返回true。

访问队列中的元素个数，如例：q.size()

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
	queue<int> q;
	for (int i=0;i<10;i++)
	{
		q.push(i);
	}

	int len = q.size();
	for (int i = 0; i < len; i++)  //注意：括号里面不能写i<q.size()因为q的长度一直在变
	{
		cout << q.front() << " ";
		q.pop();
	}

	return 0;
}

```

# deque
## 基本操作
deque容器为一个给定类型的元素进行线性处理，像向量一样，它能够快速地随机访问任一个元素，并且能够高效地插入和删除容器的尾部元素。但它又与vector不同，deque支持高效插入和删除容器的头部元素，因此也叫做双端队列。

(1)    构造函数

deque():创建一个空deque

deque(int nSize):创建一个deque,元素个数为nSize

deque(int nSize,const T& t):创建一个deque,元素个数为nSize,且值均为t

deque(const deque &):复制构造函数

(2)    增加函数

**void push_front(const T& x):双端队列头部增加一个元素X**

**void push_back(const T& x):双端队列尾部增加一个元素x**

iterator insert(iterator it,const T& x):双端队列中某一元素前增加一个元素x

void insert(iterator it,int n,const T& x):双端队列中某一元素前增加n个相同的元素x

void insert(iterator it,const_iterator first,const_iteratorlast):双端队列中某一元素前插入另一个相同类型向量的[forst,last)间的数据

(3)    删除函数

Iterator erase(iterator it):删除双端队列中的某一个元素

Iterator erase(iterator first,iterator last):删除双端队列中[first,last）中的元素

**void pop_front():删除双端队列中最前一个元素**

**void pop_back():删除双端队列中最后一个元素**

**void clear():清空双端队列中最后一个元素**

(4)    遍历函数

reference at(int pos):返回pos位置元素的引用

reference front():返回首元素的引用

reference back():返回尾元素的引用

iterator begin():返回向量头指针，指向第一个元素

iterator end():返回指向向量中最后一个元素下一个元素的指针（不包含在向量中）

reverse_iterator rbegin():反向迭代器，指向最后一个元素

reverse_iterator rend():反向迭代器，指向第一个元素的前一个元素

(5)    判断函数

bool empty() const:向量是否为空，若true,则向量中无元素

(6)    大小函数

Int size() const:返回向量中元素的个数

int max_size() const:返回最大可允许的双端对了元素数量值

(7)    其他函数

void swap(deque&):交换两个同类型向量的数据

void assign(int n,const T& x):向量中第n个元素的值设置为x

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
	deque<int> q;
	for (int i=0;i<10;i++)
	{
		q.push_back(i);
	}

	cout << q.front() << " " << *q.begin() << endl;  // front是返回首元素的引用，begin是返回首元素的指针
	int len = q.size();
	for (int i = 0; i < len; i++)  //注意：括号里面不能写i<q.size()因为q的长度一直在变
	{
		cout << q.front() << " ";
		q.pop_front();
	}
	cout << endl;
	for (int i = 0; i < 10; i++)
	{
		q.push_front(i);
	}


	len = q.size();
	for (int i = 0; i < len; i++)  //注意：括号里面不能写i<q.size()因为q的长度一直在变
	{
		cout << q.front() << " ";
		q.pop_front();
	}



	return 0;
}

```

# stack
s.push(value);   //将value压入栈
s.pop();         //将栈顶元素删除 
s.top();         //返回栈顶元素


## code 

```c++
#include <iostream>
#include<stack>
using namespace std;

int main()
{
	stack<int> s;
	for (int i = 0; i < 10; i++)
	{
		s.push(i);
	}

	int len = s.size();
	for (int i = 0; i < len; i++)  
	{
		cout << s.top() << " ";
		s.pop();
	}

	return 0;
}
```