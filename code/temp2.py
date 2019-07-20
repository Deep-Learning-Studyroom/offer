class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        head1 = pHead.next
        if head1.val != pHead.val:
            pHead.next = self.deleteDuplication(pHead.next)
        else:
            while pHead.val == head1.val and head1.next is not None:
                head1 = head1.next
            if head1.val != pHead.val:
                pHead = self.deleteDuplication(head1)
            else:
                return None
        return pHead
