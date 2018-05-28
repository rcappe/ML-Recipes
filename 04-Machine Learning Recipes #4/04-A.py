#https://www.youtube.com/watch?v=84gqSbLcBFE&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=4

# import a dataset

from sklearn import datasets
iris = datasets.load_iris()

X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split #sklearn.cross_validation is deprecaded
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.5)

# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print (predictions)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,predictions))