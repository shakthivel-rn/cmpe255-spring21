import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def load_sort():
  try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml(name='mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
  except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
  # mnist["data"], mnist["target"]
  return mnist

def sorting_by_target(mnist):
  reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
  reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
  X_train = mnist.data.iloc[reorder_train]
  y_train = mnist.target.iloc[reorder_train]
  X_test = mnist.data.iloc[reorder_test + 60000]
  y_test = mnist.target.iloc[reorder_test + 60000]
  return X_train, y_train, X_test, y_test

def train_predict(some_digit, X_train, y_train):
  sgd = sgd_train(X_train, y_train)
  return sgd.predict(pd.DataFrame(some_digit).transpose())

def sgd_train(X_train, y_train):
  shuffle_index = np.random.permutation(60000)
  X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]
  y_train_classified = (y_train == 5)
  from sklearn.linear_model import SGDClassifier
  sgd_model = SGDClassifier()
  sgd_model.fit(X_train, y_train_classified)
  return sgd_model

def calculate_cross_val_score(k, X):
  sum = 0
  for i in range(k):
    X_train, y_train, X_test, y_test = sorting_by_target(X)
    sgd_model = sgd_train(X_train, y_train)
    y_test_pred = sgd_model.predict(X_test)
    y_test_actual = (y_test == 5)
    rmse_i = calculate_rmse(pd.DataFrame(y_test_actual.to_numpy()), pd.DataFrame(y_test_pred))
    print(f"Fold {i + 1} RMSE Value - {rmse_i}")
    sum += rmse_i
  return sum / k

def calculate_rmse(Y_actual, Y_hypothesis):
  return np.sqrt(mean_squared_error(Y_actual, Y_hypothesis))

def save_fig(fig_id, tight_layout=True):
  path = "fig" + ".png"
  print("Saving figure", fig_id)
  if tight_layout:
    plt.tight_layout()
  plt.savefig(path, format='png', dpi=300)

def random_digit(X):
  some_digit = X[36000]
  some_digit_image = some_digit.reshape(28, 28)
  plt.imshow(some_digit_image, cmap=mpl.cm.binary,
             interpolation="nearest")
  plt.axis("off")

  save_fig(some_digit)
  plt.show()

if __name__ == '__main__':
  np.random.seed(42)

  # To plot pretty figures
  mpl.rc('axes', labelsize=14)
  mpl.rc('xtick', labelsize=12)
  mpl.rc('ytick', labelsize=12)

  mnist = load_sort()
  print(f"Cross Validation Score - {calculate_cross_val_score(5, mnist)}")