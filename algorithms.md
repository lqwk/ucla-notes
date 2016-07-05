# Algorithms

## Dynamic Programming

### Leetcode 62. Unique Paths

**Question:** A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid. How many possible unique paths are there?

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

**Question:** Now consider if some obstacles are added to the grids. How many unique paths would there be? An obstacle and empty space is marked as 1 and 0 respectively in the grid.

For example: There is one obstacle in the middle of a 3x3 grid as illustrated below.
```
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









