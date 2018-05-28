# https://www.youtube.com/watch?v=AoeEHqVSNOw&index=5&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal

# random algorithm
import random

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

from sklearn import datasets
iris = datasets.load_iris()

X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split #sklearn.cross_validation is deprecaded
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.5)

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print (predictions)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,predictions))