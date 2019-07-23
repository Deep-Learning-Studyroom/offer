class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

l = ListNode(1)
l.next = 2
print(l.val, l.next)