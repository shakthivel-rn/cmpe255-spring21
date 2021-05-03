# Import Libraries
from time import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

# Importing Data

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print('data loaded')
print(faces.target_names)
print(faces.images.shape)

n_samples, h, w = faces.images.shape

X = faces.data
n_features = X.shape[1]

y = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Splitting Data Into Train and Test Set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Creating a Pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# Grid Search Cross-Validation

param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005], }

gsc = GridSearchCV(
    model, param_grid
)

# Training Model
gsc = gsc.fit(X_train, y_train)

# Predicting Results

y_pred = gsc.predict(X_test)

# Evaluation Metrics

print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
print(cm)

# Heat Map

sns.heatmap(cm/np.sum(cm), fmt='.2%', cmap='Blues', annot=True)
plt.savefig('HeatMap')

# 4x6 Subplots Of Images 

def plot_gallery(images, titles, h, w, n_row=6, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        x = titles[i].split('\n')
        y = x[0].split(':')
        # print(y)
        if y[0] != 'predicted':
          plt.xticks.color = 'Red'
          plt.yticks.color = 'Red'
          plt.axes.labelcolor = 'red'
          # plt.xaxis.label.set_color('red')
          plt.tick_params(axis='x', colors='red')
          print(titles[i])
        plt.xticks(())
        plt.yticks(())
        plt.savefig('FacePlot')


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    # print("\x1b[31m\aabc\x1b[0m")
    if pred_name == true_name : 
      return 'predicted: %s\nTrue:      %s' % (pred_name, true_name)
    else:
      return 'Wrong prediction: %s\nTrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)


