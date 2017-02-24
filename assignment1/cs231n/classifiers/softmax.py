import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  scores = X.dot(W)

  #to avoide numerical instability
  scores -= np.max(scores, axis=1).reshape(scores.shape[0],1)

  unnormalized_probablities = np.exp(scores)

  probablities = unnormalized_probablities / np.sum(unnormalized_probablities, axis=1).reshape((unnormalized_probablities.shape[0], 1))

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  num_train = X.shape[0]
  num_classes = np.max(y) + 1

  for i in range(X.shape[0]):
    loss += -1 * np.log(probablities[i, y[i]])

  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  for i in range(num_train):
    for j in range(num_classes):
      del_L = probablities[i, j] if y[i] != j else probablities[i, j] - 1
      for k in range(X.shape[1]):
        dW[k, j] += X[i, k] * del_L

  dW /= num_train
  dW += reg * W

  return loss, dW


def onehotencoding(y):
  y_onehot = np.zeros((y.shape[0], np.max(y)+1))
  y_onehot[np.arange(y.shape[0]), y] = 1
  return y_onehot

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)

  #to avoide numerical instability
  scores -= np.max(scores, axis=1).reshape(scores.shape[0],1)

  unnormalized_probablities = np.exp(scores)

  probablities = unnormalized_probablities / np.sum(unnormalized_probablities, axis=1).reshape((unnormalized_probablities.shape[0], 1))

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  y_onehot = onehotencoding(y)
  loss = -1 *  np.sum(np.log(np.sum(probablities * y_onehot, axis=1))) / X.shape[0]
  loss += 0.5 * reg * np.sum(W*W)

  del_L = probablities - y_onehot
  dW = X.T.dot(del_L) / X.shape[0] + reg * W

  return loss, dW

