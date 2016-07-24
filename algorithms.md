# Algorithms

* [NUMBERS](#numbers)
  * [[E] Leetcode 204. Count Primes](#e-leetcode-204-count-primes)
* [GREEDY ALGORITHMS](#greedy-algorithms)
  * [[M] Leetcode 122. Best Time to Buy and Sell Stock II](#m-leetcode-122-best-time-to-buy-and-sell-stock-ii)
  * [[H] Leetcode 330. Patching Array](#h-leetcode-330-patching-array)
  * [[M] Leetcode 376. Wiggle Subsequence](#m-leetcode-376-wiggle-subsequence)
* [DYNAMIC PROGRAMMING](#dynamic-programming)
  * [[M] Leetcode 53. Maximum Subarray](#m-leetcode-53-maximum-subarray)
  * [[M] Leetcode 152. Maximum Product Subarray](#m-leetcode-152-maximum-product-subarray)
  * [[M] Leetcode 62. Unique Paths](#m-leetcode-62-unique-paths)
  * [[M] Leetcode 63. Unique Paths II](#m-leetcode-63-unique-paths-ii)
  * [[M] Leetcode 64. Minimum Path Sum](#m-leetcode-64-minimum-path-sum)
  * [[E] Leetcode 70. Climbing Stairs](#e-leetcode-70-climbing-stairs)
  * [[M] Leetcode 95. Unique Binary Search Trees II](#m-leetcode-95-unique-binary-search-trees-ii)
  * [[M] Leetcode 96. Unique Binary Search Trees](#m-leetcode-96-unique-binary-search-trees)
  * [[M] Leetcode 121. Best Time to Buy and Sell Stock](#m-leetcode-121-best-time-to-buy-and-sell-stock)
  * [[H] Leetcode 123. Best Time to Buy and Sell Stock III](#h-leetcode-123-best-time-to-buy-and-sell-stock-iii)
  * [[H] Leetcode 188. Best Time to Buy and Sell Stock IV](#h-leetcode-188-best-time-to-buy-and-sell-stock-iv)
  * [[H] Leetcode 174. Dungeon Game](#h-leetcode-174-dungeon-game)
  * [[M] Leetcode 279. Perfect Squares](#m-leetcode-279-perfect-squares)
  * [[M] Leetcode 338. Counting Bits](#m-leetcode-338-counting-bits)
  * [[M] Leetcode 343. Integer Break](#m-leetcode-343-integer-break)

## NUMBERS

### [E] Leetcode 204. Count Primes

[Leetcode Source](https://leetcode.com/problems/count-primes/)

**Question:**

>Count the number of prime numbers less than a non-negative number, **`n`**.

**Answer:**

Use the [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) method to find the answer.

```python
class Solution(object):

    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """

        prime = [True for x in range(n)]

        for i in range(2, n):
            if i*i >= n:
                break
            if not prime[i]:
                continue
            for j in range(i*i, n, i):
                prime[j] = False

        count = 0
        for i in range(2, n):
            if prime[i]:
                count = count+1

        return count
```


## GREEDY ALGORITHMS

### [M] Leetcode 122. Best Time to Buy and Sell Stock II

[Leetcode Source](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

**Question:**

>Say you have an array for which the `i`th element is the price of a given stock on day `i`.
>
>Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

**Answer:**

Use a greedy approach. We want to buy in the stocks at a low price and sell at a high price and find the consecutive sum of all the profits. For example, if the prices are `1 2 3 1 4`, we buy in at `1`, sell at `3`, then buy in at `1`, sell at `4`.

```python
class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        n = len(prices)
        if n == 0 or n == 1:
            return 0

        profit = 0

        for i in range(1, n):
            if prices[i] > prices[i-1]:
                profit += prices[i] - prices[i-1]

        return profit
```

### [H] Leetcode 330. Patching Array

[Leetcode Source](https://leetcode.com/problems/patching-array/)

**Question:**

> Given a sorted positive integer array `nums` and an integer `n`, add/patch elements to the array such that any number in range `[1, n]` inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required.
> 
> ```
> Example 1:
> nums = [1, 3], n = 6
> Return 1.
> ```
> 
> Combinations of `nums` are `[1], [3], [1,3]`, which form possible sums of: `1, 3, 4`.
> Now if we add/patch `2` to `nums`, the combinations are: `[1], [2], [3], [1,3], [2,3], [1,2,3]`.
> Possible sums are `1, 2, 3, 4, 5, 6`, which now covers the range `[1, 6]`.
> So we only need `1` patch.
> 
> ```
> Example 2:
> nums = [1, 5, 10], n = 20
> Return 2.
> The two patches can be [2, 4].
> ```
> 
> ```
> Example 3:
> nums = [1, 2, 2], n = 5
> Return 0.
> ```

**Answer:**

Using a greedy approach, let `miss` be the smallest sum in `[0,n]` that we might be missing. Meaning we already know we can build all sums in `[0,miss)`. Then if we have a number `num <= miss` in the given array, we can add it to those smaller sums to build all sums in `[0,miss+num)`. If we don't, then we must add such a number to the array, and it's best to add `miss` itself, to maximize the reach.

Example: Let's say the input is `nums = [1, 2, 4, 13, 43]` and `n = 100`. We need to ensure that all sums in the range `[1,100]` are possible.

Using the given numbers `1`, `2` and `4`, we can already build all sums from `0` to `7`, i.e., the range `[0,8)`. But we can't build the sum 8, and the next given number (`13`) is too large. So we insert `8` into the array. Then we can build all sums in `[0,16)`.

Do we need to insert `16` into the array? No! We can already build the sum `3`, and adding the given `13` gives us sum `16`. We can also add the `13` to the other sums, extending our range to `[0,29)`.

And so on. The given `43` is too large to help with sum `29`, so we must insert `29` into our array. This extends our range to `[0,58)`. But then the `43` becomes useful and expands our range to [0,101). At which point we're done.

```python
class Solution(object):

    def minPatches(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: int
        """

        miss, added, i = 1, 0, 0
        length = len(nums)

        while miss <= n:
            if i < length and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                miss += miss
                added += 1

        return added
```

### [M] Leetcode 376. Wiggle Subsequence

[Leetcode Source](https://leetcode.com/problems/wiggle-subsequence/)

**Question:**

> A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with fewer than two elements is trivially a wiggle sequence.
> 
> For example, `[1,7,4,9,2,5]` is a wiggle sequence because the differences `(6,-3,5,-7,3)` are alternately positive and negative. In contrast, `[1,4,7,2,5]` and `[1,7,4,5,5]` are not wiggle sequences, the first because its first two differences are positive and the second because its last difference is zero.
> 
> Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence. A subsequence is obtained by deleting some number of elements (eventually, also zero) from the original sequence, leaving the remaining elements in their original order.
> 
> Examples:
> ```
> Input: [1,7,4,9,2,5]
> Output: 6
> The entire sequence is a wiggle sequence.
> ```
> 
> ```
> Input: [1,17,5,10,13,15,10,5,16,8]
> Output: 7
> There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].
> ```
> 
> ```
> Input: [1,2,3,4,5,6,7,8,9]
> Output: 2
> ```
> 
> Follow up: Can you do it in O(n) time?

**Answer:**

```python
class Solution(object):

    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if not nums:
            return 0

        n = len(nums)

        ans, i, j = 1, 1, 0

        while i < n:
            if nums[j] < nums[i]:
                ans += 1
                while i+1 < n and nums[i+1] >= nums[i]:
                    i += 1
            elif nums[j] > nums[i]:
                ans += 1
                while i+1 < n and nums[i+1] <= nums[i]:
                    i += 1
            i, j = i+1, i

        return ans
```


## DYNAMIC PROGRAMMING

### [M] Leetcode 53. Maximum Subarray

[Leetcode Source](https://leetcode.com/problems/maximum-subarray/)

**Question:**

>Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
>
>For example, given the array `[−2,1,−3,4,−1,2,1,−5,4]`,
the contiguous subarray `[4,−1,2,1]` has the largest `sum = 6`.

**Answer:**
```python
class Solution(object):

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0

        maxsum, temp = nums[0], 0

        for num in nums:
            if temp > 0:
                temp = temp + num
            else:
                temp = num
            maxsum = max(temp, maxsum)

        return maxsum
```


### [M] Leetcode 152. Maximum Product Subarray

[Leetcode Source](https://leetcode.com/problems/maximum-product-subarray/)

**Question:**

>Find the contiguous subarray within an array (containing at least one number) which has the largest product.
>
>For example, given the array `[2,3,-2,4]`,
the contiguous subarray `[2,3]` has the largest `product = 6`.

**Answer:**

We need to keep track of both `maxprod` and `minprod` since multiplying two negative numbers will give a positive one.

Let `f(k)` be `maxprod` and `g(k)` be `minprod`, then we have

```
f(k) = max( A[k], f(k-1) * A[k], A[k], g(k-1) * A[k] )
g(k) = min( A[k], g(k-1) * A[k], A[k], f(k-1) * A[k] )
```

Other than that, we have tto check that the `num` we multiply with is not `0`.

```python
class Solution(object):

    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0

        maxprod = minprod = res = nums[0]

        for i in range(1, n):
            temp = maxprod
            maxprod = max(max(temp * nums[i], minprod * nums[i]), nums[i])
            minprod = min(min(temp * nums[i], minprod * nums[i]), nums[i])
            res = max(maxprod, res)

        return res
```

### [M] Leetcode 62. Unique Paths

[Leetcode Source](https://leetcode.com/problems/unique-paths/)

**Question:**

>A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid. How many possible unique paths are there? (**Note:** You can only move either down or right at any point in time.)

**Answer:**
```python
class Solution(object):

    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """

        paths = [[None for x in range(n)] for y in range(m)]

        for i in range(m):
            paths[i][0] = 1
        for j in range(n):
            paths[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                paths[i][j] = paths[i-1][j] + paths[i][j-1]

        return paths[m-1][n-1]
```

### [M] Leetcode 63. Unique Paths II

[Leetcode Souce](https://leetcode.com/problems/unique-paths-ii/)

**Question:**

>Now consider if some obstacles are added to the grids. How many unique paths would there be? An obstacle and empty space is marked as 1 and 0 respectively in the grid. (**Note:** You can only move either down or right at any point in time.)
>
> For example: There is one obstacle in the middle of a 3x3 grid as illustrated below.
> ```python
> [
>   [0,0,0],
>   [0,1,0],
>   [0,0,0]
> ]
> ```
> The total number of unique paths is 2.

**Answer:**
```python
class Solution(object):

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        paths = [[None for x in range(n)] for y in range(m)]
        if (obstacleGrid[0][0] == 1) or (obstacleGrid[m-1][n-1] == 1):
            paths[0][0] = 0
        else:
            paths[0][0] = 1

        for i in range(1, m):
            if (paths[i-1][0] == 0) or (obstacleGrid[i][0] == 1):
                paths[i][0] = 0
            else:
                paths[i][0] = 1

        for j in range(1, n):
            if (paths[0][j-1] == 0) or (obstacleGrid[0][j] == 1):
                paths[0][j] = 0
            else:
                paths[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    paths[i][j] = 0
                else:
                    paths[i][j] = paths[i-1][j] + paths[i][j-1]

        return paths[m-1][n-1]
```

### [M] Leetcode 64. Minimum Path Sum

[Leetcode Source](https://leetcode.com/problems/minimum-path-sum/)

**Question:**

>Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path. (**Note:** You can only move either down or right at any point in time.)

**Answer:**
```python
class Solution(object):

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        m = len(grid)
        n = len(grid[0])

        paths = [[0 for x in range(n)] for y in range(m)]
        paths[0][0] = grid[0][0]

        for i in range(1, m):
            paths[i][0] = paths[i-1][0] + grid[i][0]
        for j in range(1, n):
            paths[0][j] = paths[0][j-1] + grid[0][j]

        for i in range(1, m):
            for j in range(1, n):
                if paths[i-1][j] < paths[i][j-1]:
                    paths[i][j] = paths[i-1][j] + grid[i][j]
                else:
                    paths[i][j] = paths[i][j-1] + grid[i][j]

        return paths[m-1][n-1]
```

### [E] Leetcode 70. Climbing Stairs

[Leetcode Source](https://leetcode.com/problems/climbing-stairs/)

**Question:**

>You are climbing a stair case. It takes `n` steps to reach to the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Answer:**
```python
class Solution(object):
    
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n == 1:
            return 1
        if n == 2:
            return 2

        s = [1] * (n+1)
        s[2] = 2

        for i in range(3, n+1):
            s[i] = s[i-2] + s[i-1]

        return s[n]
```

### [M] Leetcode 95. Unique Binary Search Trees II

[Leetcode Source](https://leetcode.com/problems/unique-binary-search-trees-ii/)

**Question:**

> Given an integer `n`, generate all structurally unique BST's (binary search trees) that store values `1...n`.
> 
> For example,
> Given `n = 3`, your program should return all 5 unique BST's shown below.
> 
> ```
>    1         3     3      2      1
>     \       /     /      / \      \
>      3     2     1      1   3      2
>     /     /       \                 \
>    2     1         2                 3
> ```

**Answer:**

Generate each subtree recursively using the **depth-first-serach** algorithm.

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:

    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """

        return self.dfs(1, n) if n >= 1 else []

    def dfs(self, s, e):
        if s > e:  return [None]

        trees = []
        for i in range(s, e+1):
            ltrees = self.dfs(s, i-1)
            rtrees = self.dfs(i+1, e)
            for left in ltrees:
                for right in rtrees:
                    root = TreeNode(i)
                    root.left, root.right = left, right
                    trees.append(root)

        return trees
```

### [M] Leetcode 96. Unique Binary Search Trees

[Leetcode Source](https://leetcode.com/problems/unique-binary-search-trees/)

**Question:**

> Given `n`, how many structurally unique BST's (binary search trees) that store values `1...n`?
> 
> For example,
> Given `n = 3`, there are a total of 5 unique BST's.
> 
> ```
>    1         3     3      2      1
>     \       /     /      / \      \
>      3     2     1      1   3      2
>     /     /       \                 \
>    2     1         2                 3
> ```

**Answer:**

We enumerate through each possible `root = 1 ~ n`. Through each iteration, the left subtree can have `left = root - 1` possible roots and the right subtree can have `right = n-1 - left` possible roots. Thus the overall combination of each `root` would be `trees[left] * trees[right]`.

```python
class Solution(object):

    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """

        trees = [0] * (n+1)
        if n < 2:
            return 1
        elif n == 2:
            return 2

        trees[0] = 1
        trees[1] = 1
        trees[2] = 2

        for n in range(3, n+1):
            for left in range(0, n):
                right = n-1 - left
                trees[n] += trees[left] * trees[right]

        return trees[n]
```

### [M] Leetcode 121. Best Time to Buy and Sell Stock

[Leetcode Source](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

**Question:**

> Say you have an array for which the ith element is the price of a given stock on day `i`.
> 
> If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
> 
> Example 1:
> ```
> Input: [7, 1, 5, 3, 6, 4]
> Output: 5
> 
> max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
> ```
> 
> Example 2:
> ```
> Input: [7, 6, 4, 3, 1]
> Output: 0
> 
> In this case, no transaction is done, i.e. max profit = 0.
> ```

**Answer:**

There is a simple `O(n^2)` solution for this problem as shown below.
```python
class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        n = len(prices)
        profit = 0
        for i in range(1, n):
            for j in range(i):
                if prices[i] - prices[j] > profit:
                    profit = prices[i] - prices[j]

        return profit
```

However, we need an `O(n)` algorithm, so we use dynamic programming as shown below.
```python
class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        n = len(prices)
        if n == 0 or n == 1:
            return 0

        profit, current = 0, prices[0]

        for i in range(n):
            profit = max(profit, prices[i] - current)
            current = min(prices[i], current)

        return profit
```

### [H] Leetcode 123. Best Time to Buy and Sell Stock III

[Leetcode Source](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

**Question:**

> Say you have an array for which the `i`th element is the price of a given stock on day `i`.
> 
> Design an algorithm to find the maximum profit. You may complete at most two transactions.
> 
> **Note:** You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

**Answer:**

We first do a scan forward similar to the one we did for the single transaction one and record the max profit for each step `i` in an array. Then we reverse scan the profits array and prices array to determine if there should be a second transaction and what it is.

For example, we have the prices `1 5 2 6 6 8`, the constructed profits array is `[0, 4, 4, 5, 5, 7]`. In the reverse scan we find that we can have `profits[2] + prices[5] - prices[2]` to be the maximum profit, which is `10`.

```python
class Solution(object):

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        n = len(prices)
        if n == 0 or n == 1:
            return 0

        profits, current, profit = [], prices[0], 0
        
        for i in range(n):
            profit = max(profit, prices[i] - current)
            current = min(current, prices[i])
            profits.append(profit)

        i, result, current, profit = n-1, profits[n-1], prices[n-1], 0
        while i >= 0:
            profit = max(profit, current - prices[i])
            current = max(current, prices[i])
            result = max(result, profits[i] + profit)
            i -= 1

        return result
```

### [H] Leetcode 188. Best Time to Buy and Sell Stock IV

[Leetcode Source](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

**Question:**

> Say you have an array for which the `i`th element is the price of a given stock on day `i`.
> 
> Design an algorithm to find the maximum profit. You may complete at most `k` transactions.
> 
> **Note:** You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

**Answer:**

If we have `k >= n/2`, then it is the same as having as many transactions we want, so we use the greedy algorithm we developed above.

Otherwise, we let `profits[i][j]` be the maximum profit if we made the `i`th transaction on the `j`th day. We have

* `maxTemp = max(maxTemp, profits[i-1][j-1] – prices[j])`
    - This is the temporary maximum profit such that if we buy stock in on the `j`th day, we subtract the cost of the current purchase, which is `prices[j]` from the previous profits, which is `profits[i-1][j-1]`, accounting for one less transaction (`i-1`).
* `profits[i][j] = max(profits[i][j-1], prices[j] + maxTemp)` 
    - Since we already accounted for the purchase price in `maxTemp`, we simply add `prices[j]` to `maxTemp` to get the profit.

```python
class Solution(object):

    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """

        n = len(prices)

        if k >= (n >> 1): # fancy/fast for n/2
            return self.maxProfitGreedy(prices)

        profits =[[0 for j in range(n)] for i in range(k+1)]
 
        for i in range(1, k+1):

            maxTemp = (-prices[0])

            for j in range(1, n):
                profits[i][j] = max(profits[i][j-1], prices[j] + maxTemp)
                maxTemp = max(maxTemp, profits[i-1][j-1] - prices[j])

        return profits[k][n-1]


    def maxProfitGreedy(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        result = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                result += prices[i] - prices[i-1]

        return result
```

### [H] Leetcode 174. Dungeon Game

[Leetcode Source](https://leetcode.com/problems/dungeon-game/)

**Question:**

>The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.
>
>The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.
>
>Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).
>
>In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
>
>**Write a function to determine the knight's minimum initial health so that he is able to rescue the princess.**

**Answer:**

At first glance, this problem bears a large resemblance to the "**Maximum/Minimum Path Sum**" problem. However, a path with maximum overall health gain does not guarantee the minimum initial health, since it is essential in the current problem that the health never drops to zero or below. For instance, consider the following two paths:
```
0 -> -300 -> 310 -> 0
```
and
```
0 -> -1 -> 2 -> 0
```
The net health gain along these paths are `-300 + 310 = 10` and `-1 + 2 = 1`, respectively. The first path has the greater net gain, but it requires the initial health to be at least 301 in order to balance out the -300 loss in the second room, whereas the second path only requires an initial health of 2.

Fortunately, this problem can be solved through a **table-filling Dynamic Programming technique**, similar to other "grid walking" problems:

To begin with, we should maintain a 2D array D of the same size as the dungeon, where `D[i][j]` represents the minimum health that guarantees the survival of the knight for the rest of his quest BEFORE entering room `(i, j)`. Obviously `D[0][0]` is the final answer we are after. Hence, for this problem, we need to fill the table from the bottom right corner to left top.

Then, let us decide what the health should be at least when leaving room `(i, j)`. There are only two paths to choose from at this point: `(i+1, j)` and `(i, j+1)`. Of course we will choose the room that has the smaller `D` value, or in other words, the knight can finish the rest of his journey with a smaller initial health. Therefore we have:

```python
min_HP_on_exit = min(D[i+1][j], D[i][j+1])
```

Now `D[i][j]` can be computed from `dungeon[i][j]` and `min_HP_on_exit` based on one of the following situations:

* If `dungeon[i][j] == 0`, then nothing happens in this room; the knight can leave the room with the same health he enters the room with, i.e. `D[i][j] = min_HP_on_exit`.

* If `dungeon[i][j] < 0`, then the knight must have a health greater than `min_HP_on_exit` before entering `(i, j)` in order to compensate for the health lost in this room. The minimum amount of compensation is "`-dungeon[i][j]`", so we have `D[i][j] = min_HP_on_exit - dungeon[i][j]`.

* If `dungeon[i][j] > 0`, then the knight could enter `(i, j)` with a health as little as `min_HP_on_exit - dungeon[i][j]`, since he could gain "`dungeon[i][j]`" health in this room. However, the value of `min_HP_on_exit - dungeon[i][j]` might drop to 0 or below in this situation. When this happens, we must clip the value to 1 in order to make sure `D[i][j]` stays positive: `D[i][j] = max(min_HP_on_exit - dungeon[i][j], 1)`.

Notice that the equation for `dungeon[i][j] > 0` actually covers the other two situations. We can thus describe all three situations with one common equation, i.e.:

```python
D[i][j] = max(min_HP_on_exit - dungeon[i][j], 1)
```

for any value of `dungeon[i][j]`.

Take `D[0][0]` and we are good to go. Also, like many other "**table-filling**" problems, the 2D array D can be replaced with a 1D "rolling" array here.

```python
class Solution(object):

    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        
        m = len(dungeon)
        n = len(dungeon[0])

        paths = [[0 for x in range(n)] for y in range(m)]
        paths[m-1][n-1] = max(1, 1-dungeon[m-1][n-1])

        for i in range(m-2, -1, -1):
            paths[i][n-1] = max(1, paths[i+1][n-1] - dungeon[i][n-1])
        for j in range(n-2, -1, -1):
            paths[m-1][j] = max(1, paths[m-1][j+1] - dungeon[m-1][j])

        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                right = max(1, paths[i+1][j] - dungeon[i][j])
                down  = max(1, paths[i][j+1] - dungeon[i][j])
                paths[i][j] = min(right, down)

        return paths[0][0]
```

### [M] Leetcode 279. Perfect Squares

[Leetcode Source](https://leetcode.com/problems/perfect-squares/)

**Question:**

>Given a positive integer `n`, find the least number of perfect square numbers (for example, `1, 4, 9, 16, ...`) which sum to `n`.
>
>For example, given `n = 12`, return `3` because `12 = 4 + 4 + 4`; given `n = 13`, return `2` because `13 = 4 + 9`.

**Answer:**
```python
class Solution(object):

    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n == 0:
            return 1

        num = [sys.maxint for x in range(n+1)]
        num[0] = 0

        for i in range(1, n+1):
            tmp = num[i]
            for j in range(1, n+1):
                if j**2 > i:
                    break
                tmp = min(tmp, 1+num[i-j**2])
            num[i] = tmp

        return num[-1]
```

### [M] Leetcode 338. Counting Bits

[Leetcode Source](https://leetcode.com/problems/counting-bits/)

**Question:**

>Given a non negative integer number num. For every numbers `i` in the range `0 <= i <= num` calculate the number of 1's in their binary representation and return them as an array.
>
>Example: For `num = 5` you should return `[0,1,1,2,1,2]`.
>
>Follow up: It is very easy to come up with a solution with run time `O(n*sizeof(integer))`. But can you do it in linear time `O(n)` possibly in a single pass? Space complexity should be `O(n)`.
Can you do it like a boss? Do it without using any builtin function like `__builtin_popcount` in C++ or in any other language.
>
>Hints:
>* You should make use of what you have produced already.
>* Divide the numbers in ranges like `[2-3]`, `[4-7]`, `[8-15]` and so on. And try to generate new range from previous.
>* Or does the odd/even status of the number help you in calculating the number of 1's?

**Answer:**
```python
class Solution(object):

    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """

        ones = [0 for x in range(num+1)]
        ones[0] = 0

        if num == 0:
            return ones
        if num == 1:
            ones[1] = 1
            return ones
        if num == 2:
            ones[1] = 1
            ones[2] = 1
            return ones

        ones[1] = 1
        ones[2] = 1

        power = 2

        for count in range(3, num+1):
            if count == power*2:
                power = power*2
                ones[count] = 1
                continue
            else:
                diff = count - power
                ones[count] = 1 + ones[diff]

        return ones
```

### [M] Leetcode 343. Integer Break

[Leetcode Source](https://leetcode.com/problems/integer-break/)

**Question:**

>Given a positive integer `n`, break it into the sum of at least two positive integers and maximize the product of those integers. Return the maximum product you can get.
>
>For example, given `n = 2`, return 1 `(2 = 1 + 1)`; given `n = 10`, return 36 `(10 = 3 + 3 + 4)`.
>
>Note: You may assume that `n` is not less than 2 and not larger than 58.
>
> Hints:
> * There is a simple `O(n)` solution to this problem.
> * You may check the breaking results of `n` ranging from 7 to 10 to discover the regularities.

**Answer:**

**This algorithm uses dynamic programming, and is an `O(n)` algorithm.**

For a number `num >= 4`, we can separate it into the sum of `num-2` and `2` or the sum of `num-3` and `3`. We compare the products `products[num-2] * 2` and `products[num-3] * 3` to determine which is larger and htat is what we put in for `products[num]`. Thus this gives the dymamic programming approach.

```python
class Solution(object):

    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n <= 3:
            return n-1

        products = [1] * (n+1)
        products[2] = 2
        products[3] = 3

        for i in range(4, n+1):
            products[i] = max(products[i-2] * 2, products[i-3] * 3)

        return products[n]
```

**This algorithm uses math, and is an `O(1)` algorithm.**

This algorithm is given by a pattern found.

```python
class Solution(object):

    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n == 1:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 4

        mod = n % 3
        if mod == 0:
            return 3**(n//3)
        elif mod == 1:
            return 4 * (3**(n//3 - 1))
        elif mod == 2:
            return 2 * (3**(n//3))
```

