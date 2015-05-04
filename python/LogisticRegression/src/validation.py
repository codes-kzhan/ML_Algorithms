import warnings
import numbers

import numpy as np
import scipy.sparse as sp


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.
    Checks the sparse format of spmatrix and converts if necessary.
    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.
    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if accept_sparse is None:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    sparse_type = spmatrix.format
    if dtype is None:
        dtype = spmatrix.dtype
    if sparse_type in accept_sparse:
        # correct type
        if dtype == spmatrix.dtype:
            # correct dtype
            if copy:
                spmatrix = spmatrix.copy()
        else:
            # convert dtype
            spmatrix = spmatrix.astype(dtype)
    else:
        # create new
        spmatrix = spmatrix.asformat(accept_sparse[0]).astype(dtype)
    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    return spmatrix


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def _shape_repr(shape):
    """Return a platform independent reprensentation of an array shape
    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: "
                         "%s" % str(uniques))


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def check_array(array, accept_sparse=None, dtype="numeric", order=None, copy=False,
                force_all_finite=True, ensure_2d=True, allow_nd=False,
                ensure_min_samples=1, ensure_min_features=1):
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is converted to an at least 2nd numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.
    dtype : string, type or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    if sp.issparse(array):
        if dtype == "numeric":
            dtype = None
        array = _ensure_sparse_format(array, accept_sparse, dtype, order,
                                      copy, force_all_finite)
    else:
        if ensure_2d:
            array = np.atleast_2d(array)
        if dtype == "numeric":
            if hasattr(array, "dtype") and array.dtype.kind == "O":
                # if input is object, convert to float.
                dtype = np.float64
            else:
                dtype = None
        array = np.array(array, dtype=dtype, order=order, copy=copy)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. Expected <= 2" %
                             array.ndim)
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required."
                             % (n_samples, shape_repr, ensure_min_samples))


    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required."
                             % (n_features, shape_repr, ensure_min_features))
    return array


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


def check_X_y(X, y, accept_sparse=None, dtype="numeric", order=None, copy=False,
              force_all_finite=True, ensure_2d=True, allow_nd=False,
              multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False):
    """Input validation for standard estimators.
    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y. For multi-label y,
    set multi_output=True to allow 2d and sparse y.
    If the dtype of X is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.
    y : nd-array, list or sparse matrix
        Labels.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.
    dtype : string, type or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector.
    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.
    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y


def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error
    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.
    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().", stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})
