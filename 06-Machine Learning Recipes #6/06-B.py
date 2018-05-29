
from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow.contrib import learn

#def main(unused_argv):

#Load dataset
iris = learn.datasets.load_dataset('iris')
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris['data'], iris['taget'], test_size=0.2, random_state=42)

# Build 3 layer DNN with 10, 20 10 unit respectively.
#classifier = learn.DNNClassifier(hidden_units=[10,20,10], n_classes=2)

# Fit and predict.
#classifier.fit(x_train, y_train, steps=200)
#score = metrics.accuracy_score(y_test, classifier.predict(x_test))
#print('Accurancy: {0:f}'.format(score))
