import numpy as np
cimport numpy as np
from libc.math import sqrt, abs

cdef str argmax(dict d) except *:
	cdef str max_class = None
	cdef double max_count = 0
	cdef double total_count = 0
	cdef str key
	cdef double value
	for key, value in d.iteritems():
		total_count += value
		if value > max_count:
			max_count = value
			max_class = key
	
	return max_class


cdef str predict_max(list a) except *:
	return argmax(count_dict(a))

cdef dict count_dict(list a) except *:
	cdef dict d = {}
	cdef str x
	for x in a:
		d.setdefault(x, 0)
		d[x] += 1
	return d

cdef double mean_squared_error(list x):
	cdef np.array xnp
	xnp = np.array(x)
	xnp = xnp - xnp.mean()
	return sqrt((xnp * xnp.T).mean())

cdef double mean_absolute_error(list x):
	cdef np.array xnp
	xnp = np.array(x)
	xnp = xnp - xnp.mean()
	return abs(xnp).mean()

cdef double gini(list x):
	cdef dict d = {}
	cdef str y
	cdef double total
	cdef list to_square
	cdef np.ndarray to_square2
	for y in x:
		d.setdefault(y, 0)
		d[y] += 1
	total = len(x)
	to_square = []
	cdef str key
	cdef double value
	for key, value in d.iteritems():
		to_square.append(value/total)
	to_square2 = np.array(to_square)
	return 1 - (to_square2 * to_square.T).sum()

cdef tuple bin_split(list sample_feature, double feature_value):
	cdef np.ndarray left, right
	cdef tuple x
	left = np.array([x[1] for x in sample_feature if x[0]<=feature_value])
	right = np.array([x[1] for x in sample_feature if x[0]>feature_value])
	return left, right
