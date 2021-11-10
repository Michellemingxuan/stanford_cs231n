from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_training = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in np.arange(num_training):
        inner = X[i, :].dot(W)
        temp = np.sum(np.exp(inner))
        loss += - np.log(np.exp(X[i].dot(W[:, y[i]])) / temp)
        for j in np.arange(num_class):
            if j == y[i]:
                dW[:, j] -= X[i]
            dW[:, j] += np.exp(inner[j]) / temp * X[i]
    loss = loss / num_training + reg * np.sum(W * W)
    dW = dW / num_training + 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_training = X.shape[0]
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    mask = np.stack([np.arange(num_class)] *
                    num_training) == y.reshape((num_training, 1))
    loss = - np.sum(X.dot(W) * mask)
    loss += np.sum(np.log(np.sum(np.exp(X.dot(W)), axis=1)))
    loss = loss / num_training + reg * np.sum(W * W)
    dW = - X.T.dot(mask)
    dW += X.T.dot(np.exp(X.dot(W)) /
                  np.sum(np.exp(X.dot(W)), axis=1)[:, np.newaxis])
    dW = dW / num_training + 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
