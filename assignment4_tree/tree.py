import numpy as np
import queue
from sklearn.base import BaseEstimator

def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    
    probs = np.sum(y, axis=0) / y.shape[0] if y.shape[0] != 0 else 0
    entropy = -np.sum(probs * np.log(probs + EPS))
    return entropy
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    probs = np.sum(y, axis=0) / y.shape[0] if y.shape[0] != 0 else 0
    gini = 1 - np.sum(np.square(probs))
    return gini
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    mean = np.mean(y)
    var = np.mean(np.square(y - mean))
    return var 

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    median = np.median(y)
    mad_median = np.mean(np.abs(y- median))    
    return mad_median


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, value, proba=0):
        self.feature_index = feature_index
        self.value = value
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
class Vertex:
    """
    Class for searching in decision tree 
    """
    def __init__(self, node, X, indexes):
        self.node = node
        self.X = X
        self.indexes = indexes
    
    def get(self):
        return (self.node, self.X, self.indexes)

    
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        X_left = X_subset[X_subset[:, feature_index] < threshold]
        X_right = X_subset[X_subset[:, feature_index] >= threshold]
        y_left = y_subset[X_subset[:, feature_index] < threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """
        y_left = y_subset[X_subset[:, feature_index] < threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # assume that full search is possible here
        feature_index, threshold, g = 0, 0, np.inf
        
        for cur_feature in range(X_subset.shape[1]):
#             TODO efficiently
            for cur_threshold in np.unique(X_subset[:, cur_feature]):
                y_left, y_right = self.make_split_only_y(cur_feature, cur_threshold, X_subset, y_subset)
                cur_g = y_left.shape[0] / X_subset.shape[0] * self.criterion(y_left) + \
                y_right.shape[0] / X_subset.shape[0] * self.criterion(y_right)
                
                if cur_g < g:
                    feature_index, threshold, g = cur_feature, cur_threshold, cur_g 
                    
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset, depth):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        if depth < self.max_depth and y_subset.shape[0] > self.min_samples_split:
            feature_index, threshold = self.choose_best_split(X_subset, y_subset) 
            node = Node(feature_index=feature_index, value=threshold)
            
            (X_left, y_left), (X_right, y_right) = self.make_split(node.feature_index, node.value, X_subset, y_subset)
            node.left_child = self.make_tree(X_left, y_left, depth + 1)
            node.right_child = self.make_tree(X_right, y_right, depth + 1)
        
        else:
            #it's the case of leaf
            node = Node(None, None)
            
            if self.classification:
                #probabilities array
                #TODO
                probs = np.sum(y_subset, axis=0) / y_subset.shape[0] if y_subset.shape[0] != 0 else 0
                node.proba = probs
            else:
                node.proba = np.mean(y_subset)
                
        return node
    
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, depth=0)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        n_objects = X.shape[0]
        
        if self.classification:
            y_predicted = np.argmax(self.predict_proba(X), axis=1).reshape(n_objects)
        else:
            y_predicted = np.full(n_objects, 0.0)
            indexes = np.arange(n_objects)
            q = queue.Queue()
            q.put(Vertex(self.root, X, indexes))
            while not q.empty():
                cur_v = q.get()
                cur_node, cur_X, cur_indexes = cur_v.get()
                
                if cur_node.left_child is None and cur_node.right_child is None:
                    #leaf case
                    y_predicted[cur_indexes] = cur_node.proba
                else:
                    (X_left, y_left), (X_right, y_right) = self.make_split(cur_node.feature_index, cur_node.value, cur_X, cur_indexes)
                    q.put(Vertex(cur_node.left_child, X_left, y_left))
                    q.put(Vertex(cur_node.right_child, X_right, y_right))

        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'
        
        n_objects = X.shape[0]
        
        y_predicted_probs = np.full((n_objects, self.n_classes), 0, dtype=float)
        
        indexes = np.arange(n_objects)
        
        q = queue.Queue()
        
        q.put(Vertex(self.root, X, indexes))
        
        while not q.empty():
            cur_v = q.get()
            cur_node, cur_X, cur_indexes = cur_v.get()
            
            if cur_node.left_child is None and cur_node.right_child is None:
                #leaf case
                y_predicted_probs[cur_indexes] = cur_node.proba
            else:
                (X_left, y_left), (X_right, y_right) = self.make_split(cur_node.feature_index, cur_node.value, cur_X, cur_indexes)
                q.put(Vertex(cur_node.left_child, X_left, y_left))
                q.put(Vertex(cur_node.right_child, X_right, y_right))
        
        return y_predicted_probs
