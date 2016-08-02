# Algorithms

* [NUMBERS](#numbers)
  * [[E] Leetcode 204. Count Primes](#e-leetcode-204-count-primes)
* [STRING](#string)
  * [[E] Leetcode 344. Reverse String](#e-leetcode-344-reverse-string)
* [ARRAY](#array)
  * [[M] Leetcode 34. Search for a Range](#m-leetcode-34-search-for-a-range)
  * [[M] Leetcode 35. Search Insert Position](#m-leetcode-35-search-insert-position)
  * [[E] Leetcode 349. Intersection of Two Arrays](#e-leetcode-349-intersection-of-two-arrays)
  * [[E] Leetcode 350. Intersection of Two Arrays II](#e-leetcode-350-intersection-of-two-arrays-ii)
  * [[E] Leetcode 278. First Bad Version](#e-leetcode-278-first-bad-version)
  * [[E] Leetcode 374. Guess Number Higher or Lower](#e-leetcode-374-guess-number-higher-or-lower)
  * [[H] Leetcode 33. Search in Rotated Sorted Array](#h-leetcode-33-search-in-rotated-sorted-array)
  * [[M] Leetcode 81. Search in Rotated Sorted Array II](#m-leetcode-81-search-in-rotated-sorted-array-ii)
  * [[M] Leetcode 153. Find Minimum in Rotated Sorted Array](#m-leetcode-153-find-minimum-in-rotated-sorted-array)
  * [[H] Leetcode 154. Find Minimum in Rotated Sorted Array II](#h-leetcode-154-find-minimum-in-rotated-sorted-array-ii)
  * [[M] Leetcode 74. Search a 2D Matrix](#m-leetcode-74-search-a-2d-matrix)
  * [[M] Leetcode 240. Search a 2D Matrix II](#m-leetcode-240-search-a-2d-matrix-ii)
  * [[E] Leetcode 217. Contains Duplicate](#e-leetcode-217-contains-duplicate)
  * [[E] Leetcode 219. Contains Duplicate II](#e-leetcode-219-contains-duplicate-ii)
  * [[M] Leetcode 220. Contains Duplicate III](#m-leetcode-220-contains-duplicate-iii)
* [STACK](#stack)
  * [[M] Leetcode 71. Simplify Path](#m-leetcode-71-simplify-path)
  * [[H] Leetcode 316. Remove Duplicate Letters](#h-leetcode-316-remove-duplicate-letters)
* [GREEDY ALGORITHMS](#greedy-algorithms)
  * [[H] Leetcode 45. Jump Game II](#h-leetcode-45-jump-game-ii)
  * [[M] Leetcode 55. Jump Game](#m-leetcode-55-jump-game)
  * [[M] Leetcode 122. Best Time to Buy and Sell Stock II](#m-leetcode-122-best-time-to-buy-and-sell-stock-ii)
  * [[M] Leetcode 134. Gas Station](#m-leetcode-134-gas-station)
  * [[H] Leetcode 135. Candy](#h-leetcode-135-candy)
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
  * [[H] Leetcode 97. Interleaving String](#h-leetcode-97-interleaving-string)
  * [[M] Leetcode 121. Best Time to Buy and Sell Stock](#m-leetcode-121-best-time-to-buy-and-sell-stock)
  * [[H] Leetcode 123. Best Time to Buy and Sell Stock III](#h-leetcode-123-best-time-to-buy-and-sell-stock-iii)
  * [[H] Leetcode 188. Best Time to Buy and Sell Stock IV](#h-leetcode-188-best-time-to-buy-and-sell-stock-iv)
  * [[H] Leetcode 174. Dungeon Game](#h-leetcode-174-dungeon-game)
  * [[E] Leetcode 198. House Robber](#e-leetcode-198-house-robber)
  * [[M] Leetcode 213. House Robber II](#m-leetcode-213-house-robber-ii)
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


## STRING

### [E] Leetcode 344. Reverse String

[Leetcode Source](https://leetcode.com/problems/reverse-string/)

**Question:**

> Write a function that takes a string as input and returns the string reversed.
> Example: Given `s = "hello"`, return `"olleh"`.

**Answer:**

Use the Python stepping syntax: `[<start>:<end>:<step>]`

```python
class Solution(object):

    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """

        return s[::-1]
```


## ARRAY

### [M] Leetcode 34. Search for a Range

[Leetcode Source](https://leetcode.com/problems/search-for-a-range/)

**Question:**

> Given a sorted array of integers, find the starting and ending position of a given target value.
> 
> Your algorithm's runtime complexity must be in the order of `O(logn)`.
> 
> If the target is not found in the array, return `[-1, -1]`. For example:
> 
> ```
> Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4].
> ```

**Answer:**

The logic below is straightforward.

```python
class Solution(object):

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        n = len(nums)
        if n == 0:
            return [-1, -1]

        l = self.binarySearch(nums, target, False)
        r = self.binarySearch(nums, target, True)

        return [l, r]

    def binarySearch(self, nums, target, maxIndex):
        """
        :type nums: List[int]
        :type target: int
        :type maxIndex: boolean
        :rtype: int
        """

        left, right = 0, len(nums)-1
        index = -1

        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                index = mid
                if maxIndex:
                    left = mid+1
                else:
                    right = mid-1
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1

        return index
```

### [M] Leetcode 35. Search Insert Position

[Leetcode Source](https://leetcode.com/problems/search-insert-position/)

**Question:**

> Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
> 
> You may assume no duplicates in the array. Here are few examples:
> 
> ```
> [1,3,5,6], 5 -> 2
> [1,3,5,6], 2 -> 1
> [1,3,5,6], 7 -> 4
> [1,3,5,6], 0 -> 0
> ```

**Answer:**

```python
class Solution(object):

    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            if target <= nums[0]:
                return 0
            else:
                return 1

        left, right, mid = 0, n-1, 0
        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1

        if target > nums[mid]:
            return mid+1
        return mid
```

### [E] Leetcode 349. Intersection of Two Arrays

[Leetcode Source](https://leetcode.com/problems/intersection-of-two-arrays/)

**Question:**

> Given two arrays, write a function to compute their intersection.
> 
> Example: Given `nums1 = [1, 2, 2, 1]`, `nums2 = [2, 2]`, `return [2]`.
> 
> Note:
> * Each element in the result must be unique.
> * The result can be in any order.

**Answer:**

```python
class Solution(object):

    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        n1 = len(nums1)
        n2 = len(nums2)

        if n1 == 0 or n2 == 0:
            return []

        d, r = {}, {}

        for i in range(n1):
            num = nums1[i]
            if num not in d:
                d[num] = True

        for i in range(n2):
            num = nums2[i]
            if num in nums1 and num not in r:
                r[num] = True

        return r.keys()
```

### [E] Leetcode 350. Intersection of Two Arrays II

[Leetcode Source](https://leetcode.com/problems/intersection-of-two-arrays/)

**Question:**

> Given two arrays, write a function to compute their intersection.
> 
> Example: Given `nums1 = [1, 2, 2, 1]`, `nums2 = [2, 2]`, `return [2, 2]`.
> 
> Note:
> * Each element in the result should appear as many times as it shows in both arrays.
> * The result can be in any order.
> 
> Follow up:
> * What if the given array is already sorted? How would you optimize your algorithm?
> * What if nums1's size is small compared to nums2's size? Which algorithm is better?
> * What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

**Answer:**

```python
class Solution(object):

    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """

        n1 = len(nums1)
        n2 = len(nums2)

        if n1 == 0 or n2 == 0:
            return []

        d1, d2 = {}, {}
        r = []

        for i in range(n1):
            num = nums1[i]
            if num not in d1:
                d1[num] = 1
            else:
                d1[num] += 1

        for i in range(n2):
            num = nums2[i]
            if num not in d2:
                d2[num] = 1
            else:
                d2[num] += 1

        for key in d2.keys():
            if key in d1:
                count = min(d1[key], d2[key])
                for i in range(count):
                    r.append(key)

        return r
```

### [E] Leetcode 278. First Bad Version

[Leetcode Source](https://leetcode.com/problems/first-bad-version/)

**Question:**

> You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.
> 
> Suppose you have `n` versions `[1, 2, ..., n]` and you want to find out the first bad one, which causes all the following ones to be bad.
> 
> You are given an API `bool isBadVersion(version)` which will return whether `version` is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

**Answer:**

```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):

    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """

        left, right = 1, n
        while left < right:
            mid = (left + right) >> 1
            if isBadVersion(mid):
                right = mid
            else:
                left = mid+1

        return left
```

### [E] Leetcode 374. Guess Number Higher or Lower

[Leetcode Source](https://leetcode.com/problems/guess-number-higher-or-lower/)

**Question:**

> We are playing the Guess Game. The game is as follows:
> 
> I pick a number from `1` to `n`. You have to guess which number I picked. Every time you guess wrong, I'll tell you whether the number is higher or lower. You call a pre-defined API `guess(int num)` which returns 3 possible results (`-1`, `1`, or `0`):
> 
> ```
> -1 : My number is lower
>  1 : My number is higher
>  0 : Congrats! You got it!
> ```
> 
> Example:
> 
> ```
> n = 10, I pick 6.
> Return 6.
> ```

**Answer:**

```python
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):

    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """

        left, right = 0, n
        while left <= right:
            mid = (left + right) >> 1
            ret = guess(mid)
            if ret == 0:
                return mid
            elif ret == 1:
                left = mid+1
            elif ret == -1:
                right = mid-1
```

### [H] Leetcode 33. Search in Rotated Sorted Array

[Leetcode Source](https://leetcode.com/problems/search-in-rotated-sorted-array/)

**Question:**

> Suppose a sorted array is rotated at some pivot unknown to you beforehand.
> 
> (i.e., `0 1 2 4 5 6 7` might become `4 5 6 7 0 1 2`).
> 
> You are given a target value to search. If found in the array return its index, otherwise return `-1`.
> 
> You may assume no duplicate exists in the array.

**Answer:**

Perform binary search on the whole array and when the left number is less than or equal to the right number `nums[left] <= nums[right]`, this means we have a strictly increasing array. Then we perform binary search on this array.

```python
class Solution(object):

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return -1
        if n == 1:
            if nums[0] == target:
                return 0
            else:
                return -1

        left, right = 0, n-1
        return self.search2(nums, left, right, target)

    def search2(self, nums, left, right, target):
        """
        :type nums: List[int]
        :type left: int
        :type right: int
        :type target: int
        :rtype: int
        """

        if nums[left] <= nums[right]:
            return self.binarySearch(nums, left, right, target)

        mid = (left + right) >> 1
        ret = self.search2(nums, left, mid, target)
        if ret != -1:
            return ret

        return self.search2(nums, mid+1, right, target)

    def binarySearch(self, nums, left, right, target):
        """
        :type nums: List[int]
        :type left: int
        :type right: int
        :type target: int
        :rtype: int
        """

        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1

        return -1
```

### [M] Leetcode 81. Search in Rotated Sorted Array II

[Leetcode Source](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)

**Question:**

> Follow up for "Search in Rotated Sorted Array": What if duplicates are allowed?
> 
> Would this affect the run-time complexity? How and why?
> 
> Write a function to determine if a given target is in the array.

**Answer:**

```python
class Solution(object):

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return False
        if n == 1:
            if nums[0] == target:
                return True
            else:
                return False

        left, right = 0, n-1
        return self.search2(nums, left, right, target)

    def search2(self, nums, left, right, target):
        """
        :type nums: List[int]
        :type left: int
        :type right: int
        :type target: int
        :rtype: int
        """

        if left == right:
            return nums[left] == target

        if nums[left] < nums[right]:
            return self.binarySearch(nums, left, right, target)

        mid = (left + right) >> 1
        ret = self.search2(nums, left, mid, target)
        if ret == True:
            return ret

        return self.search2(nums, mid+1, right, target)

    def binarySearch(self, nums, left, right, target):
        """
        :type nums: List[int]
        :type left: int
        :type right: int
        :type target: int
        :rtype: int
        """

        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                return True
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1

        return False
```

### [M] Leetcode 153. Find Minimum in Rotated Sorted Array

[Leetcode Source](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

**Question:**

> Suppose a sorted array is rotated at some pivot unknown to you beforehand.
> 
> (i.e., `0 1 2 4 5 6 7` might become `4 5 6 7 0 1 2`).
> 
> Find the minimum element.
> 
> You may assume no duplicate exists in the array.

**Answer:**

```python
class Solution(object):

    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]

        left, right = 0, n-1
        return self.findMin2(nums, left, right)

    def findMin2(self, nums, left, right):
        """
        :type nums: List[int]
        :type left: int
        :type right: int
        :rtype: int
        """

        if nums[left] <= nums[right]:
            return nums[left]

        mid = (left + right) >> 1
        min1 = self.findMin2(nums, left, mid)
        min2 = self.findMin2(nums, mid+1, right)
        return min(min1, min2)
```

### [H] Leetcode 154. Find Minimum in Rotated Sorted Array II

[Leetcode Source](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

**Question:**

> Follow up for "Find Minimum in Rotated Sorted Array": What if duplicates are allowed?
> 
> Would this affect the run-time complexity? How and why?
> Suppose a sorted array is rotated at some pivot unknown to you beforehand.
> 
> (i.e., `0 1 2 4 5 6 7` might become `4 5 6 7 0 1 2`).
> 
> Find the minimum element.
> 
> The array may contain duplicates.

**Answer:**

```python
class Solution(object):

    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]

        left, right = 0, n-1
        return self.findMin2(nums, left, right)

    def findMin2(self, nums, left, right):
        """
        :type nums: List[int]
        :type left: int
        :type right: int
        :rtype: int
        """

        if left == right:
            return nums[left]

        if nums[left] < nums[right]:
            return nums[left]

        mid = (left + right) >> 1
        min1 = self.findMin2(nums, left, mid)
        min2 = self.findMin2(nums, mid+1, right)
        return min(min1, min2)
```

### [M] Leetcode 74. Search a 2D Matrix

[Leetcode Source](https://leetcode.com/problems/search-a-2d-matrix/)

**Question:**

> Write an efficient algorithm that searches for a value in an `m x n` matrix. This matrix has the following properties:
> 
> * Integers in each row are sorted from left to right.
> * The first integer of each row is greater than the last integer of the previous row.
> 
> For example:
> 
> ```
> Consider the following matrix:
> 
> [
>   [1,   3,  5,  7],
>   [10, 11, 16, 20],
>   [23, 30, 34, 50]
> ]
> 
> Given target = 3, return true.
> ```

**Answer:**

In the first method, we perform binary search first on all the rows then on a particular selected row.

```python
class Solution(object):

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        n = len(matrix)
        if n == 0:
            return False
        m = len(matrix[0])
        if m == 0:
            return False

        # binary search on the all rows
        left, right, pivot = 0, n-1, 0
        while left <= right:
            pivot = (left + right) >> 1
            if target < matrix[pivot][0]:
                right = pivot - 1
            elif target > matrix[pivot][m-1]:
                left = pivot + 1
            else:
                break

        # binary search on the selected row
        left, right = 0, m-1
        while left <= right:
            mid = (left + right) >> 1
            if target == matrix[pivot][mid]:
                return True
            if target < matrix[pivot][mid]:
                right = mid - 1
            else:
                left = mid + 1

        return False
```

In the second method, we view the 2D array as a linear array and binary search through it.

```python
class Solution(object):

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        n = len(matrix)
        if n == 0:
            return False
        m = len(matrix[0])
        if m == 0:
            return False

        left, right = 0, m*n-1
        while left <= right:
            mid = (left + right) >> 1
            if matrix[mid//m][mid%m] == target:
                return True
            elif matrix[mid//m][mid%m] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
```

### [M] Leetcode 240. Search a 2D Matrix II

[Leetcode Source](https://leetcode.com/problems/search-a-2d-matrix-ii/)

**Question:**

> Write an efficient algorithm that searches for a value in an `m x n` matrix. This matrix has the following properties:
> 
> * Integers in each row are sorted in ascending from left to right.
> * Integers in each column are sorted in ascending from top to bottom.
> 
> For example:
> 
> ```
> Consider the following matrix:
> 
> [
>   [ 1,  4,  7, 11, 15],
>   [ 2,  5,  8, 12, 19],
>   [ 3,  6,  9, 16, 22],
>   [10, 13, 14, 17, 24],
>   [18, 21, 23, 26, 30]
> ]
> 
> Given target = 5, return true.
> Given target = 20, return false.
> ```

**Answer:**

We could easily write an algorithm to binary search through each row, which will take `O(mlogn)` time, but we can do better than this. The goal is to limit the algorithm to a single step at each iteration, which means that for each comparison, we have a deterministic step.

For this I chose the bottom left as the pivot so at each iteration, we have the following conditions:

1. If `matrix[pRow][pCol] == target`, we return `True`
2. If `matrix[pRow][pCol] > target`, we move up, which is `pRow--`
3. If `matrix[pRow][pCol] < target`, we move right, which is `pCol++`

```python
class Solution(object):

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        n = len(matrix)
        if n == 0:
            return False
        m = len(matrix[0])
        if m == 0:
            return False

        # take the lower left corner to be the pivot (pRow, pCol)
        pRow, pCol = n-1, 0

        while pRow >= 0 and pCol < m:
            if target == matrix[pRow][pCol]:
                return True
            elif target < matrix[pRow][pCol]:
                pRow -= 1
            else:
                pCol += 1

        return False
```

### [E] Leetcode 217. Contains Duplicate

[Leetcode Source](https://leetcode.com/problems/contains-duplicate/)

**Question:**

> Given an array of integers, find if the array contains any duplicates. Your function should return `true` if any value appears at least twice in the array, and it should return `false` if every element is distinct.

**Answer:**

```python
class Solution(object):

    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        n = len(nums)
        if n == 0 or n == 1:
            return False

        matches = {}

        for num in nums:
            numStr = str(num)
            if numStr in matches:
                return True
            else:
                matches[numStr] = True

        return False
```

### [E] Leetcode 219. Contains Duplicate II

[Leetcode Source](https://leetcode.com/problems/contains-duplicate-ii/)

**Question:**

> Given an array of integers and an integer `k`, find out whether there are two distinct indices `i` and `j` in the array such that `nums[i] = nums[j]` and the difference between `i` and `j` is at most `k`.

**Answer:**

```python
class Solution(object):

    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """

        n = len(nums)
        if n == 0 or n == 1:
            return False

        matches = {}

        for i in range(n):
            numStr = str(nums[i])
            if numStr in matches:
                j = matches[numStr]
                if (i - j) <= k:
                    return True
            matches[numStr] = i

        return False
```

### [M] Leetcode 220. Contains Duplicate III

[Leetcode Source](https://leetcode.com/problems/contains-duplicate-iii/)

**Question:**

> Given an array of integers, find out whether there are two distinct indices `i` and `j` in the array such that the difference between `nums[i]` and `nums[j]` is at most `t` and the difference between `i` and `j` is at most `k`.

**Answer:**

We keep an `OrderedDict` so that whenever the iterator `i` goes beyond `k`, we will pop the first added item in order to save us the work of determining if the indices are out of range. Also, we bucket the numbers with the following condition:

```
abs( nums[i] - nums[j] ) <= t    <=>    abs( nums[i]/t - nums[j]/t ) <= 1
```

```python
class Solution(object):

    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """

        n = len(nums)
        if k < 1 or t < 0 or n == 0 or n == 1:
            return False

        matches = collections.OrderedDict()

        for i in range(n):
            key = nums[i] // max(1, t)

            for m in (key, key-1, key+1):
                if m in matches and abs(nums[i] - matches[m]) <= t:
                    return True

            matches[key] = nums[i]

            if i >= k:
                matches.popitem(last=False)

        return False
```

## STACK

### [M] Leetcode 71. Simplify Path

[Leetcode Source](https://leetcode.com/problems/simplify-path/)

**Question:**

> Given an absolute path for a file (Unix-style), simplify it.
> 
> For example:
> 
> ```
> path = "/home/", => "/home"
> path = "/a/./b/../../c/", => "/c"
> ```
> 
> Corner Cases:
> 
> * Did you consider the case where `path = "/../"`?
>     - In this case, you should return `"/"`.
> * Another corner case is the path might contain multiple slashes `'/'` together, such as `"/home//foo/"`.
>     - In this case, you should ignore redundant slashes and return `"/home/foo"`.

**Answer:**

```python
class Solution(object):

    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """

        ans = []
        splitPaths = path.split('/')

        for p in splitPaths:

            if p == '' or p == '.':
                continue

            if p == '..':
                if len(ans) != 0:
                    ans.pop()

            else:
                ans.append(p)

        return '/' + '/'.join(ans)
```

### [H] Leetcode 316. Remove Duplicate Letters

[Leetcode Source](https://leetcode.com/problems/remove-duplicate-letters/)

**Question:**

> Given a string which contains only lowercase letters, remove duplicate letters so that every letter appear once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.
> 
> Example:
> 
> Given `"bcabc"`
> Return `"abc"`
> 
> Given `"cbacdcbc"`
> Return `"acdb"`

**Answer:**

We use a stack approach to build up this answer. First we find the number of times a character appears in the string and store the results in `cnt`. Then we visit each character `c` in the string. Once we visit each character we decrease the number of times stored in `cnt`. Also if `c` is smaller than the character `s` in the stack and if `cnt[s]` is greater than `0`, meaning there is a remaining `s` in the string, this means that we can remove `s` from our final results, so we pop the stack and push on `c`. Each time we do this, we have to check and indicate whether `c` is currently included in the stack to prevent duplicates.

```python
class Solution(object):

    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """

        included, cnt = [False] * 26, [0] * 26
        ans = []

        for c in s:
            cnt[ord(c) - 97] += 1  # ord('a') = 97

        for c in s:
            index = ord(c) - 97
            cnt[index] -= 1

            if included[index]:
                continue

            while len(ans) != 0 and ans[-1] > c and cnt[ord(ans[-1]) - 97] != 0:
                included[ord(ans.pop()) - 97] = False

            ans.append(c)
            included[index] = True

        return ''.join(ans)
```


## GREEDY ALGORITHMS

### [H] Leetcode 45. Jump Game II

[Leetcode Source](https://leetcode.com/problems/jump-game-ii/)

**Question:**

> Given an array of non-negative integers, you are initially positioned at the first index of the array.
> 
> Each element in the array represents your maximum jump length at that position.
> 
> Your goal is to reach the last index in the minimum number of jumps.
> 
> For example:
> 
> ```
> Given array A = [2,3,1,1,4]
> The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.)
> ```
> 
> **Note:** You can assume that you can always reach the last index.

**Answer:**

We use a greedy approach. We calculate the the `curReach` (maximum reach for `jumps` number of jumps). Then we calculate `nextReach`, which is maximum reach for `jumps+1` number of jumps. When the iterator `i` reaches a range greater than `curReach`, increment `jumps` and proceed. We will finally reach the `n-1` index in `jumps` steps.

```python
class Solution(object):

    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0 or n == 1:
            return 0

        curReach, nextReach, jumps, = 0, 0, 0
        for i in range(0, n):
            if curReach < i :
                curReach = nextReach
                jumps += 1            
            nextReach = max(nextReach, nums[i]+i)

        return jumps

```

### [M] Leetcode 55. Jump Game

[Leetcode Source](https://leetcode.com/problems/jump-game/)

**Question:**

> Given an array of non-negative integers, you are initially positioned at the first index of the array.
> 
> Each element in the array represents your maximum jump length at that position.
> 
> Determine if you are able to reach the last index.
> 
> For example:
> ```
> A = [2,3,1,1,4], return true.
> A = [3,2,1,0,4], return false.
> ```

**Answer:**

Let `cover` be the indices that can be covered starting from index `0`. Each step we calculate the new `cover` index. Once we have the condition `cover >= n-1` wher `n` is the length of the array (`n-1` is the last index), we are successful and return `true`. However if we do not reach the condition at the end, we return `false`. Through the iteration, if we find ourselves without being able to reach the next step (that is we cannot cover the current index, which is `cover < i`), we also break and return `false`.

```python
class Solution(object):

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        n = len(nums)
        if n == 0 or n == 1:
            return True

        i, cover = 0, 0

        while i < n:
            if cover < i:
                break
            cover = max(cover, i+nums[i])
            if cover >= n-1:
                return True
            i += 1

        return False
```

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

### [M] Leetcode 134. Gas Station

[Leetcode Source](https://leetcode.com/problems/gas-station/)

**Question:**

> There are `N` gas stations along a circular route, where the amount of gas at station `i` is `gas[i]`.
> 
> You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from station `i` to its next station (`i+1`). You begin the journey with an empty tank at one of the gas stations.
> 
> Return the starting gas station's index if you can travel around the circuit once, otherwise return `-1`.

**Answer:**

We compute the differences and sum them up and if the total sum is less than `0`, then we cannot complete a round so we return `-1`. During each iteration, we also monitor the local sum and whenever it falls below `0`, it cannot be the `start` point. Since there is guaranteed to be a unique solution if `total >= 0`, we can find it through the greedy algorithm.

```python
class Solution(object):

    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """

        n = len(gas)
        if n == 0:
            return 0
        if n == 1:
            if gas[0] >= cost[0]:
                return 0
            else:
                return -1

        total, currentSum, start = 0, 0, 0

        for i in range(0, n):
            diff = gas[i] - cost[i]
            total += diff
            currentSum += diff
            if currentSum < 0:
                start = i+1
                currentSum = 0

        if total < 0:
            return -1
        
        return start
```

### [H] Leetcode 135. Candy

[Leetcode Source](https://leetcode.com/problems/candy/)

**Question:**

> There are `N` children standing in a line. Each child is assigned a rating value.
> 
> You are giving candies to these children subjected to the following requirements:
> 
> * Each child must have at least one candy.
> * Children with a higher rating get more candies than their neighbors.
> 
> What is the minimum candies you must give?

**Answer:**

We use a greedy approach. First initialize `candies` to have `1` for each child. We first scan from left to right, and if `ratings[i] > ratings[i-1]`, we set `candies[i] = candies[i-1]`. This will guarantee that for each child on the right, if that child has a higher rating than that on the left, he/she will have one more candy. Similarly we can from right to left to guarantee that each child on the left, if the child has a higher rating than that on the right, he/she will have one more candy. However, when we scan from right to left, we only change `candies[i]` if we have `candies[i] <= candies[i+1]`.

```python
class Solution(object):

    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """

        n = len(ratings)
        if n == 0:
            return 0
        if n == 1:
            return 1

        candies = [1] * n

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1

        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1] and candies[i] <= candies[i+1]:
                candies[i] = candies[i+1] + 1

        return sum(candies)
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
> Follow up: Can you do it in `O(n)` time?

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

### [H] Leetcode 97. Interleaving String

[Leetcode Source](https://leetcode.com/problems/interleaving-string/)

**Question:**

> Given `s1`, `s2`, `s3`, find whether `s3` is formed by the interleaving of `s1` and `s2`. For example,
> 
> ```
> Given:
> 
> s1 = "aabcc",
> s2 = "dbbca",
> 
> When s3 = "aadbbcbcac", return true.
> When s3 = "aadbbbaccc", return false.
> ```

**Answer:**

We can solve this with a simple recursive solution.

```python
class Solution(object):

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """

        if len(s1) == 0 and len(s2) == 0 and len(s3) == 0:
            return True
        if (len(s1) + len(s2)) != len(s3):
            return False

        return self.isInterleave2(s1, s2, s3)

    def isInterleave2(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """

        if len(s1) == 0 and len(s2) == 0:
            return True

        if len(s1) == 0 and len(s2) != 0:
            if s2[-1] != s3[-1]:
                return False
            else:
                return self.isInterleave2(s1, s2[:-1], s3[:-1])

        if len(s1) != 0 and len(s2) == 0:
            if s1[-1] != s3[-1]:
                return False
            else:
                return self.isInterleave2(s1[:-1], s2, s3[:-1])

        return ((s1[-1] == s3[-1]) and self.isInterleave2(s1[:-1], s2, s3[:-1])) or ((s2[-1] == s3[-1]) and self.isInterleave2(s1, s2[:-1], s3[:-1]))
```

But this exceeds the time limit because it does a lot of extra work. We can transform this into a dymanic programming algorithm that runs in `O(mn)` time, where `m` and `n` are the lengths of the strings `s1` and `s2`.

We use `interleave[i][j]` (a `boolean` value) to represent whether substring `s3[:i+j-1]` can be generated from `s1[:i]` and `s2[:j]`. In the end all we have to do is look at `interleave[n][m]`. The algorithm runs in `O(mn)` to fill in the `m x n` array.

```python
class Solution(object):

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """

        m, n = len(s1), len(s2)
        if len(s3) != m+n:
            return False

        interleave = [[False for i in range(m+1)] for j in range(n+1)]
        interleave[0][0] = True

        for i in range(1, m+1):
            interleave[0][i] = interleave[0][i-1] and (s1[i-1] == s3[i-1])
        for j in range(1, n+1):
            interleave[j][0] = interleave[j-1][0] and (s2[j-1] == s3[j-1])

        for i in range(1, m+1):
            for j in range(1, n+1):
                interleave[j][i] = (interleave[j][i-1] and (s1[i-1] == s3[i+j-1])) or (interleave[j-1][i] and (s2[j-1] == s3[i+j-1]))

        return interleave[n][m]
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

### [E] Leetcode 198. House Robber

[Leetcode Source](https://leetcode.com/problems/house-robber/)

**Question:**

> You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
> 
> Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

**Answer:**

Using dynamic programming, the two conditions at each iteration is `robbed[i-1]` or `robbed[i-2] + nums[i]`. We would like to choose the larger of the two.

```python
class Solution(object):

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])

        robbed = [0] * n
        robbed[0] = nums[0]
        robbed[1] = max(nums[0], nums[1])

        for i in range(2, n):
            robbed[i] = max(robbed[i-1], nums[i] + robbed[i-2])

        return robbed[n-1]
```

### [M] Leetcode 213. House Robber II

[Leetcode Source](https://leetcode.com/problems/house-robber-ii/)

**Question:**

> After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not get too much attention. This time, all houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, the security system for these houses remain the same as for those in the previous street.
> 
> Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

**Answer:**

We can construct two slices of the given array, one taking the first element (without the last) and one taking the last element (without the first) and perform the same dynamic programming approach as above to the two slices. Then we take the larger of the two results.

```python
class Solution(object):

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])
        if n == 3:
            return max(nums[0], max(nums[1], nums[2]))

        return max(self.rob2(nums[:-1]), self.rob2(nums[1:]))

    def rob2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])

        robbed = [0] * n

        robbed[0] = nums[0]
        robbed[1] = max(nums[0], nums[1])

        for i in range(2, n):
            robbed[i] = max(robbed[i-1], nums[i] + robbed[i-2])

        return robbed[n-1]
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

