import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, abs

ctypedef np.int_t DTYPE_t

cpdef double gini(np.ndarray y):
    cdef dict d = {}
    cdef double total
    cdef list to_square
    cdef np.ndarray to_square2
    cdef DTYPE_t i
    
    for i in y:
        d.setdefault(i, 0)
        d[i] += 1
        
    total = len(y)
    to_square = list()
    cdef double value
    cdef DTYPE_t key
    
    for key, value in d.iteritems():
        to_square.append(value/total)
        
    to_square2 = np.array(to_square)
    return 1 - (to_square2 * to_square2.T).sum()


cpdef DTYPE_t classify(np.ndarray y):
    cdef dict d = {}
    cdef DTYPE_t i
    
    for i in y:
        d.setdefault(i, 0)
        d[i] += 1
        
    cdef int Max = 0
    cdef DTYPE_t MaxValue, key
    for key in d:
        if d[key] > Max:
            Max = d[key]
            MaxValue = key
            
    return MaxValue

cpdef tuple bin_split(np.ndarray data, np.ndarray y, int feature, double value):
    cdef np.ndarray determine, data0, data1, y0, y1
    determine = data[:,feature] <= value
    data0 = data[determine]  
    data1 = data[determine == False]
    y0 = y[determine]
    y1 = y[determine == False]
    return data0, y0, data1, y1

cpdef int correctNum(y, y_pred):
    cdef int ret = 0
    cdef int t
    cdef DTYPE_t i
    
    for t, i in enumerate(y):
        if i == y_pred[t]:
            ret += 1
            
    return ret

cpdef dict probaDict(np.ndarray y):
    cdef dict ret = {}
    cdef DTYPE_t i
    cdef DTYPE_t key
    
    for i in y:
        ret.setdefault(i, 0)
        ret[i] += 1
        
    for key in ret:
        ret[key] /= float(len(y))
        
    return ret

cpdef tuple feat_map(np.ndarray data, np.ndarray y, int feat_index, set feat_set):
    cdef dict xd = dict()
    cdef dict yd = dict()
    cdef double split_val
    cdef int t, i
    cdef np.ndarray feat_array = data[:, feat_index]

    for split_val in feat_set:
        xd.setdefault(split_val, [0.0 for i in range(9)])
        yd.setdefault(split_val, 0)
    for t, i in enumerate(y):
        xd[feat_array[t]][i] += 1
        yd[feat_array[t]] += 1

    return xd, yd
