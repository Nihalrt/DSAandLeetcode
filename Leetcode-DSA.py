
from collections import deque


# Kadane's Algorithm
def kadane(A):
    max_cur = max_glo = A[0]
    for i in range(1,len(A)):
        max_cur = max(A[i], max_cur+A[i])
        max_glo = max(max_cur, max_glo)
    return max_glo

# v = kadane([1,-3,2,1,-1])
# print(v)


# -----------------------------------------------------------------------



def max_window(array, k):
    total = 0
    max_total = 0
    for i in range(len(array)-k+1):
        if i==0:
            total = sum(array[i:i+k])
        else:
            total-=array[i-1]
            if i+k-1<len(array):
                total+=array[i+k-1]
        max_total = max(total, max_total)

        
    return max_total

ans = max_window([1,2,3,42,54,2,2,1,4,5,3,5,2,2,44], 2)
print(ans)
        
def minSubArrayLen(target: int, nums: list[int]) -> int:
    total = count = start = 0
    min_count = float('inf')
    
    for i in range(len(nums)):
        total += nums[i]
        count+=1
        while total>=target:
            min_count = min(min_count, count)
            total -= nums[start]
            start+=1
            count-=1
    return min_count if min_count!=float('inf') else 0
            
        

ans = minSubArrayLen(11, [0,1,2,3,4,5,6,0,8,0,-1])
print(ans)

#------------------------------------------------------------------------

def merge(intervals: list[list[int]]) -> list[list[int]]:
    result = []
    if len(intervals)==1:
        return intervals
    intervals.sort()
    
    merged = []
    x = 0
    y = 1
    while x<len(intervals):
        merged = intervals[x]
        while y<len(intervals):
            if merged[0]<=intervals[y][0]<=merged[1] and merged[1]<=intervals[y][1]:
                merged[1] = intervals[y][1]
                y+=1
            else:
                break
        result.append(merged)
        x = y
    return result

#---------------------------------------------------------------------------------------------

def generateMatrix(n: int) -> list[list[int]]:
        n_elements = n*n
        result = [[0]*n for _ in range(n)]
        top = 0
        left = 0
        right = n-1
        bottom = n-1
        nums = 1

        while nums<=n_elements:
            for x in range(left, right+1):
                result[top][x] = nums
                nums+=1
            top+=1
            for y in range(top, right+1):
                result[y][right] = nums
                nums+=1
            right-=1
            for x in range(right, left-1, -1):
                result[right+1][x] = nums
                nums+=1
            for y in range(right, top-1, -1):
                result[y][left] = nums
                nums+=1
            left+=1
        return result
    
#--------------------------------------

def spiralOrder(matrix: list[list[int]]) -> list[int]:
        result = []
        nums = len(matrix) * len(matrix[0])
        n = 0
        left = 0
        right = len(matrix[0]) - 1
        top = 0
        k = 0

        while n<nums:
            for i in range(left, right+1):
                result.append(matrix[top][i])
                n+=1
                if n==nums:
                    return result
            top+=1
            while top<len(matrix)+k:
                result.append(matrix[top][right])
                top+=1
                n+=1
                if n==nums:
                    return result
            top-=1
            for l in range(right-1, left-1, -1):
                result.append(matrix[top][l])
                n+=1
                if n==nums:
                    return result
            top-=1
            while top>0:
                result.append(matrix[top][left])
                top-=1
                n+=1
                if n==nums:
                    return result
            left+=1
            top+=1
            k-=1
        return result
    

quite = spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# -----------------------

def uniquePaths(m: int, n: int) -> int:
        x,y,z = 1,1,1
        paths=0

        for i in range(1, (m + n-2)+1):
            x *= i
                
        for j in range(1, m):
            y*=j
            
        for k in range(1, n):
            z*=k
            
        paths = x//(y*z)
        return paths

vans = uniquePaths(3,7)

#-----------------------------------------

from math import inf
from typing import List, Optional

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def is_safe(board, row, col):
            # Check column
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            # Check left diagonal
            i, j = row, col
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            
            # Check right diagonal
            i, j = row, col
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            
            return True

        def backtrack(board, row):
            if row == n:
                # Add current board configuration to result
                result.append(["".join(r) for r in board])
                return
            
            for col in range(n):
                if is_safe(board, row, col):
                    board[row][col] = 'Q'
                    backtrack(board, row + 1)
                    board[row][col] = '.'  # Backtrack

        result = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        backtrack(board, 0)
        return result

# Example usage
sol = Solution()
print(sol.solveNQueens(4))

#---------------------------------------------------------------------

def BinarySearch(List: list[int], key: int, l:int, r:int, mid: int):
    if l>r:
        return False
    mid = (l+r)//2
    if List[mid]==key:
        return True
    if key<List[mid]:
        r = mid-1
        return BinarySearch(List,key,l,r,mid)
    if key>List[mid]:
        l = mid+1
        return BinarySearch(List, key, l, r, mid)
    return False
# mine = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# vy = BinarySearch(mine, 100, 0, len(mine)-1, 0)
# print(vy)


#--------------------------------------------------------------

def minPathSum(grid):
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = grid[0][0]
        
        # Initialize the top row
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        # Initialize the left column
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        
        # Fill up the DP table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        return dp[-1][-1]

tv = minPathSum([[1,3,1],[1,5,1],[4,2,1]])
print(tv)

#-------------------------------------------------------------

minmovescount=0
def Recur(w1,w2,movescount,canrmv,canadd):
    global minmovescount
    if movescount+abs(len(w1)-len(w2))>=minmovescount:
        return minmovescount
    if w1==w2:
        return movescount
    if w1=="":
        return min(minmovescount,len(w2)+movescount)
    if w2=="":
        return min(minmovescount,len(w1)+movescount)
    
    if w1[0]==w2[0]:
        return Recur(w1[1:],w2[1:],movescount,True,True)
    else:
        a=Recur(w1[1:],w2[1:],movescount+1,False,False)#zmieniamy pierwsza litere
        b,c=inf,inf
        minmovescount=min(minmovescount,a)
        if canrmv:
            b=Recur(w1[1:],w2,movescount+1,True,False)#odejmujemy pierwsza litere
            minmovescount=min(minmovescount,b)
        if canadd:
            c=Recur(w1,w2[1:],movescount+1,False,True)#dodajemy pierwsza litere
            minmovescount=min(minmovescount,c)
        return min(a,b,c)
    
    return inf


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        global minmovescount
        minmovescount=max(len(word1),len(word2))
        
        return Recur(word1,word2,0,True,True)
    

# answer = Solution()
# answer = answer.minDistance("horse", "ros")

# -------------------------------------------------------

def setZeroes(matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        positions = {}
        l = 0
        m = 0

        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value==0:
                    positions[(i, j)] = 0
        
        for x in positions:
            l = x[0]
            m = x[1]
            while m < len(matrix[l]):
                if matrix[l][m]!=0:
                    matrix[l][m] = 0
                m+=1
            m = x[1]
            while m >= 0:
                if matrix[l][m]!=0:
                    matrix[l][m] = 0
                m-=1
                
            m = x[1]
            while l < len(matrix):
                if matrix[l][m]!=0:
                    matrix[l][m] = 0
                l+=1
            l = x[0]
            
            while l >=0:
                if matrix[l][m]!=0:
                    matrix[l][m] = 0
                l-=1
cake = setZeroes([[1,1,1],[1,0,1],[1,1,1]])


#----------------------------------------------------

def searchMatrix(matrix: List[List[int]], target: int) -> bool:
        l, r = 0, len(matrix)-1
        mid = 0

        while  l<=r:
            mid = (l+r) // 2
            if matrix[mid][0]<target<matrix[mid][len(matrix[mid])-1] and target in matrix[mid]:
                return True
            elif matrix[l][0]<target<matrix[mid][len(matrix[l])-1]:
                r = mid - 1
            elif matrix[mid][0]<target<matrix[r][len(matrix[r])-1]:
                l = mid + 1
            else:
                return False
        return False
    

#----------------------------------

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         dummy = ListNode(0)
#         dummy.next = head
#         prev = dummy
#         current = head

#         while current:

#             while current.next and current.val==current.next.val:

#                 current = current.next

#             if prev.next==current:
#                 prev = prev.next
#             else:
#                 prev.next = current.next
            
#             current = current.next
        
#         return dummy.next

# head = ListNode(1)
# head.next = ListNode(1)
# head.next.next = ListNode(2)
# head.next.next.next = ListNode(3)
# head.next.next.next.next = ListNode(3)
# head.next.next.next.next.next = ListNode(4);
# head.next.next.next.next.next.next = ListNode(4);
# head.next.next.next.next.next.next.next = ListNode(5);



# bam = Solution()
# x = bam.deleteDuplicates(head)


#-------------------------------------------------------

# Subsets

def subsets(nums: list[int]) -> list[list[int]]:

        result = []

        def backtrack(start, sub):
            if sub!=[] and sub not in result:
                result.append(sub[:])
                return
            if sub in result:
                return
            
            for i in range(start, len(nums)):
        
                sub.append(nums[i])
                backtrack(start+1, sub)
                if len(sub)==len(nums):
                    sub.pop()
        backtrack(0, [])
        return result

ans = subsets([1,2,3])
print(ans)

# combinations 

def combine(n: int, k: int) -> List[List[int]]:
        result = []
        def backtrack(start, sub):
            if len(sub)==k:
                result.append(sub[:])
            for i in range(start, n+1):
                sub.append(i)
                backtrack(i+1, sub)
                sub.pop()
            
        backtrack(1, [])
        return result
    
f = combine(4, 2)

# Partition List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution2:
    def partition(self, head: ListNode, x: int) -> ListNode:
        less_head = ListNode(0)
        greater_head = ListNode(0)
        less = less_head
        greater = greater_head
        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                greater.next = head
                greater = greater.next
            head = head.next
        
        greater.next = None
        less.next = greater_head.next
        
        return less_head.next

# head = ListNode(1)
# head.next = ListNode(4)
# head.next.next = ListNode(3)
# head.next.next.next = ListNode(2)
# head.next.next.next.next = ListNode(5)
# head.next.next.next.next.next = ListNode(2);

# bam = Solution2()
# cram = bam.partition(head, 3)

#----------------------
def sortColors(nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        a = b = 0
        keep = 0
        while b<len(nums):
            if nums[b]<nums[a]:
                while a>0 and nums[b]<nums[a]:
                    a-=1
                if nums[a]<nums[b]:
                    a+=1
                keep = nums[b]
                del nums[b]
                nums.insert(a, keep)
                a+=1
            b+=1

sortColors([2,0,2,1,1,0]          
)

#-------------------------------------------------

def exist(board: list[list[str]], word: str) -> bool:
        def backtrack(row, col, index):
            if index==len(word):
                return True
            if index>=len(word):
                return False
            
            if row < 0 or row >= len(board) or col < 0 or col >= len(board[0]) or board[row][col]!=word[index]:
                return False
            
            temp = board[row][col]
            board[row][col] = "#"
            
            found = (
                backtrack(row, col+1,  index+1) or
                backtrack(row, col-1,index+1) or
                backtrack(row-1, col,index+1) or
                backtrack(row+1, col,index+1) 
            )
            
            board[row][col] = temp
            
            
            return found

        for i in range(0, len(board)):
            for j in range(0, len(board[0])):
                if board[i][j] == word[0] and backtrack(i, j, 0):
                    return True
        
        return False
            
        

# Merge Sort Algorithm

def merge_sort(arr: list[int]) -> None:
    if len(arr) > 1:
        left_arr = arr[:len(arr)//2]
        right_arr = arr[len(arr)//2:]
        
        #recursion
        merge_sort(left_arr)
        merge_sort(right_arr)
        
        i = j = k = 0
        
        while i < len(left_arr) and j < len(right_arr):
            
            if left_arr[i] < right_arr[j]:
                arr[k] = left_arr[i]
                i+=1
                
            else:
                arr[k] = right_arr[j]
                j+=1
            k+=1
        
        while i < len(left_arr):
            arr[k] = left_arr[i]
            i+=1
            k+=1
            
        while j < len(right_arr):
            arr[k] = right_arr[j]
            j+=1
            k+=1
    
    
        
cap = [2,5,4,2,5,6,1,0,3,-1,10]
merge_sort(cap)
print(cap)

# Split Linked Lists Into Parts
class Solution:
    def splitListToParts(self, head: ListNode, k: int):
        # Step 1: Calculate the length of the linked list
        length = 0
        current = head
        while current:
            length += 1
            current = current.next
        
        # Step 2: Determine the size of each part
        part_size = length // k
        extra_nodes = length % k
        
        # Step 3: Split the list
        parts = []
        current = head
        for i in range(k):
            part_head = current
            for j in range(part_size + (1 if i < extra_nodes else 0) - 1):
                if current:
                    current = current.next
            if current:
                next_part = current.next
                current.next = None
                current = next_part
            parts.append(part_head)
        
        return parts
    






# Finding unique possible structural BST, given the number of nodes. Contructing each and every possible tree

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        if n==0:
            return []
        
        def backtrack(start, end):
            if start > end:
                return [None]

            all_trees= []
            for i in range(start, end+1):
                left_tree = backtrack(start, i-1)
                right_tree = backtrack(i+1, end)

                for l in left_tree:
                    for r in right_tree:
                        current_tree = TreeNode(i)
                        current_tree.left = l
                        current_tree.right = r
                        all_trees.append(current_tree)
            return all_trees
            
        return backtrack(1, n)
    
Tree = Solution()
Tree = Tree.generateTrees(3)
    

# Finding unique possible structural BST, given the number of nodes.

class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        for i in range(2, n+1):
            for j in range(1, i+1):
                dp[i]+=dp[j-1] * dp[i-j]
        return dp[n] 
    



# Checking if the two trees are structurally identical 

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val!=q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """

        def inorder(node):
            if node:
                return inorder(node.left) + [node] + inorder(node.right) 
            else:
                return []

        
        nodes = inorder(root)
        x, y = None, None

        for i in range(len(nodes)-1):
            if nodes[i].val > nodes[i+1].val:
                y = nodes[i+1]
                if not x:
                    x = nodes[i]
                else:
                    break
        x.val, y.val = y.val, x.val
    

# root1 = TreeNode(1)
# root1.left = TreeNode(3)
# root1.right = TreeNode(2)

# answer = Solution()
# answer = answer.recoverTree(root1)

def mySqrt(x: int) -> int:
        if x==0 or x==1:
            return x
        
        left, right = 0, x
        while left<=right:
            mid = left + (right-left)//2
            if mid*mid==x:
                return mid
            elif mid*mid<x:
                left=mid+1
            else:
                right=mid-1
        return right

# MaxDepth Using Recursion 
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        if not root:
            return 0
        
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return max(left_depth, right_depth) + 1


# Solved the question by l and dummy as pointers. 
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head or left == right:
            return head

        dummy = ListNode(0)
        dummy.next = head
        l = dummy  
        for _ in range(left - 1):
            l = l.next

        prev, curr = None, l.next
        for _ in range(right - left + 1):  
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        l.next.next = curr  
        l.next = prev       

        return dummy.next
    

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)  
head.next.next.next.next = ListNode(5)

answer = Solution()
kap = answer.reverseBetween(head, 2, 4)


# Solved a Leetcode Easy (RemoveElement) using two pointers
def removeElement(self, nums: List[int], val: int) -> int:
        j = 0
        for i in range(0, len(nums)):
            if nums[i]!=val:
                nums[j] = nums[i]
                j+=1
        return j



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Converted a sorted ARRAY to a BST
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None

        mid = len(nums) // 2
        root = TreeNode(nums[mid])


        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1:])

        return root


# Converted a sorted LIST to a BST
def findMiddle(head):


    if not head:
        return None

    prev = None
    slow = head
    fast = head

    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next

    if prev:
        prev.next = None

    return slow
class Solution:



    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:

        if not head:

            return None

        if not head.next:

            return TreeNode(head.val)

        mid = findMiddle(head)

        root = TreeNode(mid.val)

        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(mid.next)

        return root



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# Solved is_balanced leetcode Easy
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def check_height(node):
            if not node:
                return 0

            left = check_height(node.left)
            right = check_height(node.right)

            if left==-1:
                return -1
            if right==-1:
                return -1

            if abs(left - right) > 1:
                return -1

            return max(left, right) + 1
        return check_height(root) != -1
    

# Solved mindepth easy using recursion
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)

        if not root.left or not root.right:
            return 1 + max(left, right)
        else:
            return 1 + min(left, right)
        
     
# solved PathSum problem using recursion   
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        
        if not root.left and not root.right and targetSum ==  root.val:
            return True

        targetSum -= root.val

        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)




# Solved Binary Tree Level Order Traversal Using A Queue to extract the levels
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            level = []
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
            result.append(level)

        return result
    # ZigZag level order traversal
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])
        left_to_right = True

        while queue:
            level_size = len(queue)
            level = []

            for i in range(level_size):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                
            if not left_to_right:
                level.reverse()
            
            result.append(level)
            left_to_right = not left_to_right
        return result
    
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)



# Run the test
sol = Solution()
print(sol.levelOrder(root))


# Solved the PathSum 2 using dfs recursion with backtracking 
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result = []

        def dfs(node, path, remaining):
            if not node:
                return
            
            path.append(node.val)
            remaining -= node.val

            if not node.left and not node.right and remaining == 0:
                result.append(list(path))
            
            dfs(node.left, path, remaining)
            dfs(node.right, path, remaining)

            path.pop()
        
        dfs(root, [], targetSum)
        return result
    
    
# Sovled a Buy-Sell Stock Question through temp variables:
        
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        
        min_price = float('inf')
        max_price = 0
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > max_price:
                max_price = price - min_price
                
        return max_price
    
    # Solved Grouping Strings K Daily Streak using List Comprehension
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        chunks = [
    group if len(group) == k else group + fill*(k - len(group))
    for group in (s[i:i+k] for i in range(0, len(s), k))
]

        return chunks


        
        


    
    



    

        
        
        









    



            
            
                
