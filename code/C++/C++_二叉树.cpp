/*
二叉树（熟练掌握前序遍历pre_order、中序遍历in_order、后序遍历post_order和层次遍历level_order）
满二叉树：叶子节点全部都在最底层，除了叶子节点，每个节点都有左右两个子节点；
完全二叉树：叶子节点都在最底下两层，最后一层的叶子节点都靠左排列，并且除了最后一层，其它层的节点个数都要达到最大；
二叉搜索树（Binary Search Tree）是满足如下性质的二叉树：
（1）如果左子树非空，则左子树上所有节点的值均小于根节点的值；
（2）如果右子树非空，则右子树上所有节点的值均大于根节点的值；
（3）左右子树本身又各是一棵二叉排序树。
BST的中序遍历是单调递增的。
*/

#include <iostream>
#include<queue>
#include<stack>
using namespace std;


//二叉树
#include<iostream>

struct TreeNode {
public:
	int val;
	int tag;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x)
	{
		val = x;
		tag = 0;
		left = NULL;
		right = NULL;
	}
	TreeNode()
	{
		val = NULL;
		tag = 0;
		left = NULL;
		right = NULL;
	}
};

void pre_order(TreeNode* root)
{
	if (root != NULL)
	{
		cout << root->val << " ";
		pre_order(root->left);
		pre_order(root->right);
	}
}
/*
使用非递归的方法实现先序遍历
使用栈来临时存储节点：
（1）打印根节点数据；
（2）把根节点的right入栈，遍历左子树；
（3）遍历完左子树返回时，栈顶元素应为right，出栈，遍历以该指针为跟的子树
*/
void pre_order_stack(TreeNode* root)
{
	TreeNode* p = root;
	stack<TreeNode*> s;
	while (p != NULL || !s.empty())
	{
		while (p != NULL)
		{
			cout << p->val << " ";
			s.push(p);
			p = p->left;
		}
		if (!s.empty())
		{
			p = s.top();
			s.pop();
			p = p->right;  //指向右子节点，下一次会先序遍历其左子树
		}
	}
}

void in_order(TreeNode* root)
{
	if (root != NULL)
	{
		in_order(root->left);
		cout << root->val << " ";
		in_order(root->right);
	}
}

/*
使用非递归的方法实现中序遍历
使用栈来存储临时节点，
（1）先将根节点入栈，遍历左子树；
（2）遍历完左子树返回时，栈顶元素应为根节点，此时出栈，并打印节点数据；
（3）再中序遍历右子树。
*/
void in_order_stack(TreeNode* root)
{
	TreeNode* p = root;
	stack<TreeNode*> s;

	while (p != NULL || !s.empty())
	{
		while (p != NULL)
		{
			s.push(p);
			p = p->left;
		}
		if (!s.empty())
		{
			p = s.top();
			s.pop();
			cout << p->val << " ";
			p = p->right;
		}

	}
}


void post_order(TreeNode* root)
{
	if (root != NULL)
	{
		post_order(root->left);
		post_order(root->right);
		cout << root->val << " ";
	}
}

/*
使用非递归的方法实现后序遍历
假设root时遍历树的根指针，后序遍历要求在遍历完左、右子树后再访问根，
需要判断根节点的左、右子树是否均遍历过。
使用标记法，节点入栈时，配一个标志tag一同入栈，tag为0表示遍历左子树前的现场保护，
tag为1表示遍历右子树前的现场保护。
首先将root和tag（0）入栈，遍历左子树；返回后，修改栈顶tag为1，遍历右子树，最后访问根节点。
*/
void post_order_stack(TreeNode* root)
{
	TreeNode* p = root;
	stack<TreeNode*> s;
	while (p != NULL || !s.empty())
	{
		while (p != NULL)
		{
			s.push(p);
			p = p->left;
		}
		if (!s.empty())
		{
			p = s.top();  //得到栈顶元素
			if (p->tag == 1)  //tag为1时
			{
				cout << p->val << " ";
				s.pop();  
				p = NULL;  //第二次访问标志其右子树已经遍历了
			}
			else
			{
				p->tag = 1;  
				p = p->right;  //指向右子节点，下次遍历其左子树
			}
		}
	}
}



void level_order(TreeNode* root)
{
	queue<TreeNode*> q;
	TreeNode* node = NULL;
	q.push(root);

	while (!q.empty())
	{
		node = q.front();
		q.pop();
		cout << node->val << " ";
		if (node->left != NULL)
		{
			q.push(node->left);
		}
		if (node->right != NULL)
		{
			q.push(node->right);
		}
	}

}


int main()
{
	TreeNode* p1 = &TreeNode(1);
	TreeNode* p2 = &TreeNode(2);
	TreeNode* p3 = &TreeNode(3);
	TreeNode* p4 = &TreeNode(4);
	TreeNode* p5 = &TreeNode(5);
	
	p1->left = p2;
	p1->right = p3;
	p2->left = p4;
	p2->right = p5;

	//先序遍历
	cout << "pre order\n";
	pre_order(p1);//递归的方法
	cout << endl;
	pre_order_stack(p1);//栈的方法

	//中序遍历
	cout << "\nin order\n";
	in_order(p1);
	cout << endl;
	in_order_stack(p1);

	//后序遍历
	cout << "\npost order\n";
	post_order(p1);
	cout << endl;
	post_order_stack(p1);

	//层次遍历
	cout << "\nlevel order\n";
	level_order(p1);
	cout << endl;
	
	return 0;
}