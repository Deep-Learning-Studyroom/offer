#### Example-1
```c++
// 读取数字，以空格隔开，然后输出他们的和
#include <iostream>
using namespace std;
int main() {
	int sum = 0;
	cout << "please input a list of number:";
	int i;
	while (cin >> i) {
		sum += i;
		while (cin.peek() == ' ') {  //屏蔽空格
			cin.get();
		}
		if (cin.peek() == '\n') {
			break;
		}
	}
	cout << "sum = " << sum << endl;
	return 0;
}
```

#### Example-2
>
```c++
// 输入 I love algorithms!
// 得到 algorithm。只有9个字符，原因是C++里字符串默认最后一个是0，也就是说以0结尾。
#include <iostream>
using namespace std;
int main() {
	char buf[20];
	cin.ignore(7);
	cin.getline(buf, 10);
	cout << buf << endl;
	return 0;
}
```

#### Example-3
>
```c++
// 输入一段文本，然后把它输出出来
#include <iostream>
using namespace std;
int main() {
	char p;
	cout << "请输入一段文本：\n";

	while (cin.peek() != '\n') {
		p = cin.get();
		cout << p;
	}
	cout << endl;
	return 0;
}
```

#### Example-4
```c++
// 输入一段文本，然后把它输出出来
#include <iostream>
using namespace std;
int main() {
	const int SIZE = 50;
	char buf[SIZE];

	cout << "请输入一段文本：";
	cin.read(buf, 20);

	cout << "字符串收集到的字符数为：" << cin.gcount() << endl;

	cout << "输入的文本信息是：";
	cout.write(buf, 20);
	cout << endl;
	return 0;
}
```

#### Example-5
```c++
#include <iostream>
using namespace std;
int main() {
	double result = sqrt(3.0);
	cout << "对3开方保留小数点后0~9位，结果如下：\n" << endl;
	for (int i = 0; i <= 9; i++) {
		cout.precision(i);
		cout << result << endl;
	}

	cout << "当前的输出精度为：" << cout.precision() << endl;
	return 0;
}
```

#### Example-6
```c++
#include <iostream>
using namespace std;
int main() {
	int width = 4;
	char str[20];

	cout << "请输入一段文本：\n";
	cin.width(5);

	while (cin >> str) {
		cout.width(width++);
		cout << str << endl;
		cin.width(5);
	}
	return 0;
}
```

#### Example-7
>任务：读取文件内容并打印出来（换行不打印）。

主要用到**ifstream**类，读取文件的类。
```c++
#include <iostream>
#include <fstream>
using namespace std;
int main() {
	ifstream in;
	in.open("test.txt"); //这两行可以写为 ifstream in("test.txt")
	if (!in) {
		cerr << "打开文件失败" << endl;
		return 0;
	}

	char x;
	while (in >> x) {
		cout << x;
	}
	cout << endl;
	in.close();
	return 0;
}
```

#### Example-8

>写入文件，主要用到**ofstring**类。

```c++
#include <iostream>
#include <fstream>
using namespace std;
int main() {
	ofstream out;
	out.open("test2.txt");
	if (!out) {
		cerr << "打开文件失败！" << endl;
		return 0;
	}
	for (int i = 0; i < 10; i++) {
		out << i;
	}

	out << endl;
	out.close();
	return 0;
}
```

#### Example-9
```c++
// 函数的重载
#include <iostream>

using namespace std;
int calc(int val);
int calc(int val1, int val2);
int calc(int val1, int val2, int val3);

int main() {
	int val1 = 1;
	int val2 = 2;
	int val3 = 3;

	int res1 = calc(val1);
	int res2 = calc(val1, val2);
	int res3 = calc(val1, val2, val3);

	cout << res1 << endl << res2 << endl << res3 << endl;
	return 0;
}

int calc(int val) {
	int res = val * val;
	return res;
}

int calc(int val1, int val2) {
	int res = val1 * val2;
	return res;
}

int calc(int val1, int val2, int val3) {
	int res = val1 + val2 + val3;
	return res;
}
```

#### Example-10
```c++
//定义一个容纳10个整数的数组，这些整数来自用户输入。我们将计算这些值的累加和、平均值并输出。

#include <iostream>
using namespace std;


int main() {
	const unsigned short ITEM = 4;
	int arr[ITEM];
	cout << "请输入" << ITEM << "个整形数据" << endl;
	for (int i = 0; i < ITEM; i++)
	{
		cout << "请输入第" << i + 1 << "个整数：";
		cin >> arr[i];
	}

	int total = 0;
	for (int j = 0; j < ITEM; j++)
	{
		total += arr[j];
	}
	cout << "总和是：" << total << endl;
	cout << "平均值是：" << total / (ITEM * 1.0) << endl;
	return 0;
}


```

#### Example-11
```c++
//读取字符串输入并显示

#include <iostream>
#include <string>
using namespace std;

int main() {
	string str;
	cout << "Please input a string\n";
	getline(cin, str);
	cout << str << endl;

	cout << "Please input a string again\n";
	cin >> str;  //可以看到，cin会忽略空格。所以要用上面的getline函数。
	cout << str << endl;

	return 0;
}


```

#### Example-12
```c++
// 函数传址
#include <iostream>
using namespace std;
void swap1(int x, int y);
void swap2_1(int* x, int* y);
void swap2_1(int* x, int* y);
void swap3(int& x, int& y);

int main() {
	int x = 1;
	int y = 2;
	cout << "x:" << x << " y:" << y << endl;

	swap1(x, y);
	cout << "After swap(传值):" << x << " " << y << "\n";

	x = 1;
	y = 2;
	swap2_1(&x, &y);
	cout << "After swap(传址):" << x << " " << y << "\n";

	x = 1;
	y = 2;
	swap3(x, y);
	cout << "After swap(传引用):" << x << " " << y << "\n";

	return 0;
}

void swap1(int x, int y)  // 传值
{
	x ^= y;
	y ^= x;
	x ^= y;
}

void swap2_1(int* x, int* y)  // 传址
{
	int temp;
	temp = *x;
	*x = *y;
	*y = temp;
}

void swap2_2(int* x, int* y) // 传址，用异或来交换两个数
{
	*x ^= *y;
	*y ^= *x;
	*x ^= *y;
}

void swap3(int& x, int& y)  // 传引用
{
	x ^= y;
	y ^= x;
	x ^= y;
}
```

#### Example-13 
```c++
#include <iostream>
#include <string>
using namespace std;

class Animal
{
public:
	string mouth;

	void eat();
	void sleep();
};

class Bird : public Animal
{
public:
	void fly();
};

class Turtle : public Animal
{
public:
	void swim();
};

void Animal::sleep()
{
	cout << "I am sleeping\n";
}

void Animal::eat()
{
	cout << "I am eating\n";
}

void Bird::fly()
{
	cout << "I am flying\n";
}

void Turtle::swim()
{
	cout << "I am swimming\n";
}

int main()
{
	Bird bird;
	Turtle turtle;
	bird.eat();
    turtle.eat();
	bird.fly();
	turtle.swim();
	return 0;
}
```

#### Example-14
```c++
//继承中构造函数和析构函数的调用顺序
#include <iostream>
#include <string>
using namespace std;

class BaseClass
{
public:
	int val_public;
	BaseClass();
	~BaseClass();
	void doSomething();
protected:
	int val_protected;
private:
	int val_private;

};

class SubClass : public BaseClass
{
public:
	SubClass();
	~SubClass();
};

BaseClass::BaseClass()
{
	cout << "进入基类构造器\n";
}

BaseClass::~BaseClass()
{
	cout << "进入基类析构器\n";
}

void BaseClass::doSomething()
{
	cout << "进入基类doSomething函数\n";
}

SubClass::SubClass()
{
	cout << "进入子类构造器\n";
}
SubClass::~SubClass()
{
	cout << "进入子类析构器\n";
}
int main()
{
	SubClass subclass;
	subclass.doSomething();
	subclass.val_public = 10;
	cout << "**********\n";
	BaseClass baseclass;
	baseclass.val_public = 10;
	return 0;
}
```

#### Example-15
```c++

```

