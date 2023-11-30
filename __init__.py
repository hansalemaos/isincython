import os
import subprocess
import sys
import numpy as np


def _dummyimport():
    import Cython


try:
    from .sort2 import fast_isin_cython, isin_cython_string
except Exception as e:
    cstring = r"""# distutils: language=c++
# distutils: extra_compile_args=/openmp
# distutils: extra_link_args=/openmp
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=True
# cython: overflowcheck.fold=False
# cython: embedsignature=False
# cython: embedsignature.format=c
# cython: cdivision=True
# cython: cdivision_warnings=False
# cython: cpow=True
# cython: c_api_binop_methods=True
# cython: profile=False
# cython: linetrace=False
# cython: infer_types=False
# cython: language_level=3
# cython: c_string_type=bytes
# cython: c_string_encoding=default
# cython: type_version_tag=True
# cython: unraisable_tracebacks=False
# cython: iterable_coroutine=True
# cython: annotation_typing=True
# cython: emit_code_comments=False
# cython: cpp_locals=False

cimport cython
import numpy as np
cimport numpy as np
import cython
from cython.parallel cimport prange
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport array    
from libc.stdio cimport printf
ctypedef fused real:
    cython.bint
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.float
    cython.double
    cython.longdouble
    cython.size_t
    cython.Py_ssize_t
    cython.Py_hash_t
    cython.Py_UCS4
    


cpdef void fast_isin_cython(real[:] arr, real[:] bigarr,np.npy_bool[:] resultarray):
    cdef unordered_set[real] mySet
    
    cdef Py_ssize_t i
    cdef Py_ssize_t arrlen = arr.shape[0]
    cdef Py_ssize_t bigarrlen=bigarr.shape[0]
    cdef Py_ssize_t key_to_check
    mySet.reserve(arrlen)
    cdef unordered_set[real].iterator it
    with nogil:
        for i in range(arrlen):
            mySet.insert(arr[i])
    for key_to_check in prange(bigarrlen,nogil=True):
        it = mySet.find( bigarr[key_to_check])
        if it != mySet.end():
            resultarray[key_to_check]=True


cdef void vector_sort_cython_stringsub(vector[string] my_vector,unordered_set[string] mySet,np.npy_bool[:]resultarray, Py_ssize_t lbigarr):
    cdef Py_ssize_t key_to_check
    cdef unordered_set[string].iterator it
    for key_to_check in prange(lbigarr,nogil=True):
        it = mySet.find( my_vector[key_to_check])
        if it != mySet.end():
            resultarray[key_to_check]=True

cpdef void isin_cython_string(needels, haystack, cython.int lneedels, cython.int lhaystack ,np.npy_bool[:] resultarray):
    cdef vector[string] my_vector 
    cdef unordered_set[string] mySet

    cdef Py_ssize_t lneedelsp=lneedels
    cdef Py_ssize_t lhaystackp=lhaystack
    cdef Py_ssize_t i
    for i in range(lhaystackp):
            my_vector.push_back(haystack[i])
    for i in range(lneedelsp):
            mySet.insert(needels[i])    
    vector_sort_cython_stringsub(my_vector,mySet,resultarray[:],lhaystackp)
"""
    pyxfile = f"sort2.pyx"
    pyxfilesetup = f"sort2compiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'sort2', 'sources': ['sort2.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='fastuniq',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .sort2 import fast_isin_cython, isin_cython_string
    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()

stringtypes = ["S", "a"]


def generate_random_arrays(shape, dtype="float64", low=0, high=1):
    return np.random.uniform(low, high, size=shape).astype(dtype)


def isincython(haystack, needels):
    resu = np.zeros(haystack.shape, dtype=bool)
    fast_isin_cython(needels, haystack, resu)
    return resu


def stri_isin(haystack, needels):
    stri = needels
    stri2 = haystack
    if stri.dtype.char not in ["S", "a"]:
        stri = stri.astype("S")

    if stri2.dtype.char not in ["S", "a"]:
        stri2 = stri2.astype("S")

    resu = np.zeros(stri2.shape[0] + 4, dtype=bool)
    lenorigstri = len(stri)
    lenorigstri2 = len(stri2)
    isin_cython_string(stri, stri2, lenorigstri, lenorigstri2, resu)
    return resu[:-4]


def fast_isin(haystack, needels):
    try:
        if haystack.dtype.char == "U":
            haystack = haystack.astype("S")

        if needels.dtype.char == "U":
            needels = needels.astype("S")

        if haystack.dtype.char in stringtypes or needels.dtype.char in stringtypes:
            return stri_isin(haystack, needels)
        return isincython(haystack, needels)
    except Exception as fe:
        sys.stderr.write(f"{fe} - Trying it with NumPy")
        sys.stderr.flush()
        return np.isin(haystack, needels)
