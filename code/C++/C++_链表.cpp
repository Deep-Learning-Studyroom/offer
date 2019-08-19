#include <iostream>
using namespace std;

struct node  //定义一个链表节点
{
	int data;  //节点的值
	node* next; //下一个节点
};

int length(node* head)  //求链表的长度
{
	int len = 1;
	node* p;
	p = head->next;
	while (p != NULL)
	{
		len++;
		p = p->next;
	}
	return len;
}

void print(node* head)  //打印链表
{
	node* p;
	p = head;
	if (p == NULL)
	{
		cout << "打印结束\n";
	}
	while (p != NULL)
	{
		cout << p->data << " ";
		p = p->next;
	}
	cout << endl;
}

//查找单链表pos位置的节点，返回节点指针
//pos从0开始，0返回head节点的指针
node* search_node(node* head, int pos)
{
	node* p = head;
	if (pos < 0)
	{
		cout << "Wrong pos value!\n";
		return NULL;
	}
	if (pos == 0)
	{
		return head;
	}
	while (pos--)
	{
		p = p->next;
		if (p == NULL)
		{
			cout << "Incorrect position to search node\n";
		}
	}
	return p;
}

//在单链表pos位置处插入节点，返回链表头指针
//pos从0开始计数，0表示插入到head节点的前面（即插入的节点是第pos个)
node* insert_node(node* head, int pos, int data)
{
	node* item = new node;
	item->data = data;
	
	if (pos == 0)
	{
		item->next = head;
		return item;
	}
	node* p;
	p = search_node(head, pos - 1);
	item->next = p->next;
	p->next = item;
	return head;
}

//单链表节点的删除
node* delete_node(node* head, int pos)
{
	if (pos == 0)
	{
		return head->next;
	}
	
	node* p_before = search_node(head, pos-1);
	node* p = search_node(head, pos);
	if (p->next != NULL)
	{
		node* p_after = p->next;
		p_before->next = p_after;
	}
	else
	{
		p_before->next = NULL;
	}
	
	return head;

}

//反转链表
node* reverse_node(node* head)
{
	//空链表或只有一个节点
	if (head == NULL || head->next == NULL)
	{
		return head;
	}
	//只有两个节点
	else if (head->next->next == NULL)
	{
		head->next->next = head;
		node* new_head = head->next;
		head->next = NULL;
		return new_head;
	}
	//多于两个节点
	else
	{
		node* node_b = NULL;
		node* node_cur = head;
		node* node_n = head->next;
		while (node_cur -> next != NULL)
		{
			node_n = node_cur->next;
			node_cur->next = node_b;
			node_b = node_cur;
			node_cur = node_n;
		}
		node_cur->next = node_b;
		return node_cur;
	}
}

//寻找链表的中间元素
//

int main()
{
	node* p1= new node;
	p1->data = 1;
	node* p2 = new node;
	p1->next = p2;
	p2 ->data = 2;
	p2->next = NULL;

	//链表的长度
	int len = 0;
	len = length(p1);
	cout << len << "\n";
	//打印链表
	print(p1);
	
	//链表中搜索第k个节点
	cout << search_node(p1, 1)->data << "\n";
	
	//链表中插入节点
	node* head = insert_node(p1, 0, 100);
	head = insert_node(head, 2, 300);
	print(head);

	//删除链表中的某个节点
	head = delete_node(head, 2);
	print(head);

	//反转链表
	head = reverse_node(head);
	print(head);
	
	
	//删除动态内存
	delete p1;
	delete p2;
	p1 = NULL;
	p2 = NULL;
	return 0;
}
