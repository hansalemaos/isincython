# fast isin() function using Cython (C++) - up to 80 times faster than NumPy/Pandas.

## pip install isincython

### Tested against Python 3.11 / Windows 10



## Cython (and a C/C++ compiler) must be installed to use the optimized Cython implementation.

This module provides functions for efficiently checking if elements in one array
are present in another array. It includes a Cython implementation for improved performance.

Note: The Cython implementation is compiled during the first import, and the compiled
extension module is stored in the same directory. Subsequent imports will use the
precompiled module for improved performance.



```python
import timeit
from isincython import generate_random_arrays, fast_isin
import numpy as np

size = 10000000
low = 0
high = 254
arras = [
    (size, "float32", low, high),
    (size, "float64", low, high),
    (size, np.uint8, low, high),
    (size, np.int8, low, high),
    (size, np.int16, low, high),
    (size, np.int32, low, high),
    (size, np.int64, low, high),
    (size, np.uint16, low, high),
    (size, np.uint32, low, high),
    (size, np.uint64, low, high),
]

reps = 1
for a in arras:
    arr = generate_random_arrays(*a)
    seq = generate_random_arrays(size // 10, *a[1:])
    s = """u=fast_isin(arr,seq)"""
    u = fast_isin(arr, seq)
    print("c++", arr[u])
    t1 = timeit.timeit(s, globals=globals(), number=reps) / reps
    print(t1)
    s2 = """q=np.isin(arr,seq)"""
    q = np.isin(arr, seq)
    print("numpy", arr[q])

    t2 = timeit.timeit(s2, globals=globals(), number=reps) / reps
    print(t2)
    print(np.all(q == u))

    print("-----------------")

haystack = np.array(
    [
        b"Cumings",
        b"Heikkinen",
        b"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        b"aaa",
        b"bbbb()",
        b"Futrelle",
        b"Allen",
        b"Cumings, Mrs. John Bradley (Florence Briggs Thayer)q",
        b"Braund, Mr. Owen Harris",
        b"Heikkinen, Miss. Laina",
        b"Futrelle, Mrs. Jacques Heath (Lily May Peel)",
        b"Allen, Mr. William Henry",
        b"Braund",
    ],
    dtype="S",
)
needels = np.array(
    [
        b"Braund, Mr. Owen Harris",
        b"Cumings, Mrs. John Bradley (Florence Briggs Th",
        b"Heikkinen, Miss. Lxxaina",
        b"Futrelle, Mrs. Jacqxues Heath (Lily May Peel)",
        b"Allen, Mxr. William Henry",
        b"sdfsdd",
        b"aaa",
        b"bbbb()",
    ],
    dtype="S",
)
haystack = np.ascontiguousarray(np.concatenate([haystack for _ in range(200000)]))
needels = np.ascontiguousarray(np.concatenate([needels for _ in range(10000)]))

s = "o = fast_isin(haystack, needels)"
t1 = timeit.timeit(s, globals=globals(), number=reps) / reps
s1 = "o = np.isin(haystack, needels)"
t2 = timeit.timeit(s1, globals=globals(), number=reps) / reps
print(f"c++ {t1}")
print(f"numpy {t2}")
o1 = fast_isin(haystack, needels)
o2 = np.isin(haystack, needels)
print(np.all(o1 == o2))
needels = needels.astype("U")
haystack = haystack.astype("U")
s = "o = fast_isin(haystack, needels)"
t1 = timeit.timeit(s, globals=globals(), number=reps) / reps
s1 = "o = np.isin(haystack, needels)"
t2 = timeit.timeit(s1, globals=globals(), number=reps) / reps
print(f"c++ {t1}")
print(f"numpy {t2}")
o1 = fast_isin(haystack, needels)
o2 = np.isin(haystack, needels)
print(np.all(o1 == o2))

# c++ [136.03264   62.5741   156.39038  ...  78.545906 229.14676  186.44472 ]
# 0.39614199999778066
# numpy [136.03264   62.5741   156.39038  ...  78.545906 229.14676  186.44472 ]
# 2.1623376999996253
# True
# -----------------
# c++ []
# 0.4184691000045859
# numpy []
# 2.189824300003238
# True
# -----------------
# c++ [126 128  31 ... 113 190 146]
# 0.011114299995824695
# numpy [126 128  31 ... 113 190 146]
# 0.05381579999811947
# True
# -----------------
# c++ [  23   35   52 ...   54   98 -125]
# 0.010347299998102244
# numpy [  23   35   52 ...   54   98 -125]
# 0.8121466000011424
# True
# -----------------
# c++ [144  29  89 ...  90  34 202]
# 0.012101899999834131
# numpy [144  29  89 ...  90  34 202]
# 0.05841199999849778
# True
# -----------------
# c++ [ 93  51 131 ... 231 147 140]
# 0.013264799999888055
# numpy [ 93  51 131 ... 231 147 140]
# 0.07822610000584973
# True
# -----------------
# c++ [138 158 233 ...  64  82 160]
# 0.018734699995548
# numpy [138 158 233 ...  64  82 160]
# 0.09425780000310624
# True
# -----------------
# c++ [158  17 126 ...  55   7 116]
# 0.011595800002396572
# numpy [158  17 126 ...  55   7 116]
# 0.06014610000420362
# True
# -----------------
# c++ [ 60  12 226 ... 152 190 155]
# 0.013999900002090726
# numpy [ 60  12 226 ... 152 190 155]
# 0.07416449999436736
# True
# -----------------
# c++ [239  84  81 ... 146  85  63]
# 0.026196500002697576
# numpy [239  84  81 ... 146  85  63]
# 0.11476380000385689
# True
# -----------------
# c++ 0.7991062000000966
# numpy 2.1993997000026866
# True
# c++ 1.7051588000031188
# numpy 3.0464809000040987
# True

```