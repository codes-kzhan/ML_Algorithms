from abc import ABCMeta
import numpy as np
import scipy.sparse as sp

def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    return meta("NewBase", bases, {})

class BaseEstimator(object):
    """Base class for all estimators in scikit-learn
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

class BaseRandomProjection(with_metaclass(ABCMeta, BaseEstimator,
                                          TransformerMixin)):
    """Base class for random projections.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, n_components='auto', eps=0.1, dense_output=False,
                 random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.dense_output = dense_output
        self.random_state = random_state

        self.components_ = None
        self.n_components_ = None

    @abstractmethod
    def _make_random_matrix(n_components, n_features):
        """ Generate the random projection matrix
        Parameters
        ----------
        n_components : int,
            Dimensionality of the target projection space.
        n_features : int,
            Dimensionality of the original source space.
        Returns
        -------
        components : numpy array or CSR matrix [n_components, n_features]
            The generated random matrix.
        """

    def fit(self, X, y=None):
        """Generate a sparse random projection matrix
        Parameters
        ----------
        X : numpy array or scipy.sparse of shape [n_samples, n_features]
            Training set: only the shape is used to find optimal random
            matrix dimensions based on the theory referenced in the
            afore mentioned papers.
        y : is not used: placeholder to allow for usage in a Pipeline.
        Returns
        -------
        self
        """
        X = check_array(X, accept_sparse=['csr', 'csc'])

        n_samples, n_features = X.shape

        if self.n_components == 'auto':
            self.n_components_ = johnson_lindenstrauss_min_dim(
                n_samples=n_samples, eps=self.eps)

            if self.n_components_ <= 0:
                raise ValueError(
                    'eps=%f and n_samples=%d lead to a target dimension of '
                    '%d which is invalid' % (
                        self.eps, n_samples, self.n_components_))

            elif self.n_components_ > n_features:
                raise ValueError(
                    'eps=%f and n_samples=%d lead to a target dimension of '
                    '%d which is larger than the original space with '
                    'n_features=%d' % (self.eps, n_samples, self.n_components_,
                                       n_features))
        else:
            if self.n_components <= 0:
                raise ValueError("n_components must be greater than 0, got %s"
                                 % self.n_components_)

            elif self.n_components > n_features:
                warnings.warn(
                    "The number of components is higher than the number of"
                    " features: n_features < n_components (%s < %s)."
                    "The dimensionality of the problem will not be reduced."
                    % (n_features, self.n_components),
                    DataDimensionalityWarning)

            self.n_components_ = self.n_components

        # Generate a projection matrix of size [n_components, n_features]
        self.components_ = self._make_random_matrix(self.n_components_,
                                                    n_features)

        # Check contract
        assert_equal(
            self.components_.shape,
            (self.n_components_, n_features),
            err_msg=('An error has occurred the self.components_ matrix has '
                     ' not the proper shape.'))

        return self

    def transform(self, X, y=None):
        """Project the data by using matrix product with the random matrix
        Parameters
        ----------
        X : numpy array or scipy.sparse of shape [n_samples, n_features]
            The input data to project into a smaller dimensional space.
        y : is not used: placeholder to allow for usage in a Pipeline.
        Returns
        -------
        X_new : numpy array or scipy sparse of shape [n_samples, n_components]
            Projected array.
        """
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.components_ is None:
            raise NotFittedError('No random projection matrix had been fit.')

        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                'Impossible to perform projection:'
                'X at fit stage had a different number of features. '
                '(%s != %s)' % (X.shape[1], self.components_.shape[1]))

        X_new = safe_sparse_dot(X, self.components_.T,
                                dense_output=self.dense_output)
        return X_new


class GaussianRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through Gaussian random projection
    The components of the random matrix are drawn from N(0, 1 / n_components).
    Parameters
    ----------
    n_components : int or 'auto', optional (default = 'auto')
        Dimensionality of the target projection space.
        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.
        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.
    eps : strictly positive float, optional (default=0.1)
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'.
        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.
    random_state : integer, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        matrix at fit time.
    Attributes
    ----------
    n_component_ : int
        Concrete number of components computed when n_components="auto".
    components_ : numpy array of shape [n_components, n_features]
        Random matrix used for the projection.
    See Also
    --------
    SparseRandomProjection
    """
    def __init__(self, n_components='auto', eps=0.1, random_state=None):
        super(GaussianRandomProjection, self).__init__(
            n_components=n_components,
            eps=eps,
            dense_output=True,
            random_state=random_state)

    def _make_random_matrix(self, n_components, n_features):
        """ Generate the random projection matrix
        Parameters
        ----------
        n_components : int,
            Dimensionality of the target projection space.
        n_features : int,
            Dimensionality of the original source space.
        Returns
        -------
        components : numpy array or CSR matrix [n_components, n_features]
            The generated random matrix.
        """
        random_state = check_random_state(self.random_state)
        return gaussian_random_matrix(n_components,
                                      n_features,
                                      random_state=random_state)
