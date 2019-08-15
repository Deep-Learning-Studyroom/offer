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

```