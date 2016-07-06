# Algorithms

* [Numbers](#numbers)
    - [Leetcode 204. Count Primes](#leetcode-204-count-primes)
* [Dynamic Programming](#dynamic-programming)
    - [Leetcode 62. Unique Paths](#leetcode-62-unique-paths)
    - [Leetcode 63. Unique Paths II](#leetcode-63-unique-paths-ii)
    - [Leetcode 64. Minimum Path Sum](#leetcode-64-minimum-path-sum)
    - [Leetcode 174. Dungeon Game](#leetcode-174-dungeon-game)
    - [Leetcode 279. Perfect Squares](#leetcode-279-perfect-squares)
    - [Leetcode 338. Counting Bits](#leetcode-338-counting-bits)

## NUMBERS

### Leetcode 204. Count Primes

[Leetcode Source](https://leetcode.com/problems/count-primes/)

**Question:** Count the number of prime numbers less than a non-negative number, **`n`**.

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


## DYNAMIC PROGRAMMING

### Leetcode 62. Unique Paths

[Leetcode Source](https://leetcode.com/problems/unique-paths/)

**Question:** A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid. How many possible unique paths are there? (**Note:** You can only move either down or right at any point in time.)

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

### Leetcode 63. Unique Paths II

[Leetcode Souce](https://leetcode.com/problems/unique-paths-ii/)

**Question:** Now consider if some obstacles are added to the grids. How many unique paths would there be? An obstacle and empty space is marked as 1 and 0 respectively in the grid. (**Note:** You can only move either down or right at any point in time.)

For example: There is one obstacle in the middle of a 3x3 grid as illustrated below.
```python
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
```
The total number of unique paths is 2.

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

### Leetcode 64. Minimum Path Sum

[Leetcode Source](https://leetcode.com/problems/minimum-path-sum/)

**Question:** Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path. (**Note:** You can only move either down or right at any point in time.)

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

### Leetcode 174. Dungeon Game

[Leetcode Source](https://leetcode.com/problems/dungeon-game/)

**Question:** The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).

In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

**Write a function to determine the knight's minimum initial health so that he is able to rescue the princess.**

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

### Leetcode 279. Perfect Squares

[Leetcode Source](https://leetcode.com/problems/perfect-squares/)

**Question:** Given a positive integer `n`, find the least number of perfect square numbers (for example, `1, 4, 9, 16, ...`) which sum to `n`.

For example, given `n = 12`, return `3` because `12 = 4 + 4 + 4`; given `n = 13`, return `2` because `13 = 4 + 9`.

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

### Leetcode 338. Counting Bits

[Leetcode Source](https://leetcode.com/problems/counting-bits/)

**Question:** Given a non negative integer number num. For every numbers `i` in the range `0 ≤ i ≤ num` calculate the number of 1's in their binary representation and return them as an array.

Example: For `num = 5` you should return `[0,1,1,2,1,2]`.

Follow up: It is very easy to come up with a solution with run time `O(n*sizeof(integer))`. But can you do it in linear time `O(n)` possibly in a single pass? Space complexity should be `O(n)`.
Can you do it like a boss? Do it without using any builtin function like `__builtin_popcount` in C++ or in any other language.

Hints:
* You should make use of what you have produced already.
* Divide the numbers in ranges like `[2-3]`, `[4-7]`, `[8-15]` and so on. And try to generate new range from previous.
* Or does the odd/even status of the number help you in calculating the number of 1's?

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




