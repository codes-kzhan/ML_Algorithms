import numpy as np
from .validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


class LogisticRegression:
    """Logistic regression using sgd optimizer.
    Parameters
    ----------
    reg (float): regularization parameter to prevent overfitting
    tolerance (float): early exit for optimizer
    verbose (boolean): set to false if no debug/progress info is to be displayed
    epoch (int): times of training rounds
    learning_rate (float): rate of gradient descent optimization
    batch_size (int): size of each mini-batch
    X (ndarray):  (m, n) dimension data points
    y (ndarray):  n target labels
    
    References:
    -----------
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
    """

    def __init__(self, reg = 0.0001, tolerance = 1e-4,
                 verbose = True, epoch = 20,
                 learning_rate = 0.01, batch_size = 200):
        """Initialize the LogisticRegression class object.
        Parameters:
        ----------
        reg (float): regularization parameter to prevent overfitting
        tolerance (float): early exit for optimizer
        verbose (boolean): set to false if no debug/progress info is to be displayed
        epoch (int): times of training rounds
        learning_rate (float): rate of gradient descent optimization
        batch_size (int): size of each mini-batch
        """
        self.reg = reg
        self.tolerance = tolerance
        self.alpha = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.le = LabelEncoder()
        self.lb = LabelBinarizer()
        self.theta_unroll = 0
        self.has_trained_once = False

    def train(self, X, y):
        """The main function, to fit the dataset
        Parameters
        ----------
        X (ndarray):  (m, n) dimension data points
        y (ndarray):  n target labels
        
        Outputs
        ----------
        use model to fit the dataset
        """
        epoch = self.epoch
        reg = self.reg
        alpha = self.alpha
        batch_size = self.batch_size
        
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64, order="C")
        classes = np.unique(y)
        self.num_classes = len(classes)
        self.predict_dtype = y.dtype

        m, k = np.shape(X)
        self.num_features = k
        
        self._initialize_theta()

        self.le.fit(y)
        lbin = self.lb.fit_transform(y)

        idx = np.random.permutation(range(m))
        X = X[idx]
        lbin = lbin[idx]
        theta = self.theta_unroll
        num_batch = int(m / self.batch_size)
        
        for i in range(epoch):
            for j in range(num_batch):
                X_batch = X[i*batch_size : (i+1)*batch_size]
                lbin_batch = lbin[i*batch_size : (i+1)*batch_size]
                cost, grad = self._cost(theta, self.num_classes, reg,
                                        X_batch, lbin_batch)
                theta = theta - alpha * grad
            if self.verbose:
                print "epoch #%d\t cost = %f\n" % (i+1, cost)
        self.theta_unroll = theta

    def _initialize_theta(self):
        """Initialize the model weights
        Outputs
        ----------
        the initial model weights
        """
        if self.has_trained_once == False:
            self.has_trained_once = True
            r = np.sqrt(6.) / np.sqrt(self.num_features + self.num_classes + 1)
            theta_unroll = np.random.rand(self.num_classes *
                                    self.num_features) * 2 * r - r
            self.theta_unroll = np.array(theta_unroll, dtype='float')

    def _cost(self, theta_unroll, num_classes, reg, X, lbin):
        """Compute the cost and grad of each mini-batch
        Parameters
        ----------
        theta_unroll (ndarray): current weights
        num_classes (int): number of unique classes of labels(y)
        reg (float): regularization parameter to prevent overfitting
        X (ndarray):  (m, n) dimension data points
        lbin (ndarray): a matrix of binarized labels(y)
        
        Outputs
        ----------
        cost: the cost of the current model
        grad: the grad to adapt the weights
        """
        m, k = np.shape(X)
        theta = np.reshape(theta_unroll, (k, num_classes))
        M = X.dot(theta)
        M_max = M.max(axis=1)
        M = np.transpose(M) - M_max
        M = np.exp(M)
        M /= M.sum(0)
        M = np.transpose(M)
        cost = np.sum(np.log(M)*lbin) / (-1. * m) + np.sum(theta*theta) * reg * 0.5
        grad = np.transpose(X).dot(lbin-M) / (-1. * m) + reg * theta
        grad = np.reshape(grad, k*num_classes)
        
        return cost, grad

    def predict(self, X):
        """Make predictions
        Parameters
        ----------
        X (ndarray):  (m, n) dimension data points
        
        Outputs
        ----------
        predict (ndarray): the prediction made by model
        prob (ndarray): the probability prediction
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        m, k = np.shape(X)
        theta = np.reshape(self.theta_unroll, (k, self.num_classes))

        M = X.dot(theta)
        M_max = M.max(axis=1)
        M = np.transpose(M) - M_max
        M = np.exp(M)
        M /= M.sum(0)
        predict_raw = np.argmax(M, axis=0)
        prob = np.max(M, axis=0)
        predict = self.le.inverse_transform(predict_raw)
        
        return predict, prob

    def score(self, X, y):
        """Compute the accuracy of the model
        Parameters
        ----------
        X (ndarray):  (m, n) dimension data points
        y (ndarray):  n target labels

        Outputs
        ----------
        score (float): accuracy of the model
        """
        predict, _ = self.predict(X)
        ret = 0
        for t, i in enumerate(predict):
            if y[t] == i:
                ret += 1
        return float(ret) / len(y)
        
