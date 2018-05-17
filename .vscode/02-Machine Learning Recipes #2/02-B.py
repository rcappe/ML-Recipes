import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0 , 50, 100] #first of eatch flower types

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))

# Visualization code
# Install 
# - pydot using this command "pip install -U pydotplus"
# - Stable 2.38 Windows install packages http://www.graphviz.org/download/
#
# !!! ISSUE !!!
# pydotplus.graphviz.InvocationException: GraphViz's executables not found
#
# on file C:\Users\myuser\AppData\Local\Programs\Python\Python36\Lib\site-packages\pydotplus\graphviz.py
# remove at line 628 this statement -< for path in (
# add near the line 628 +> path = r"C:\Program Files (x86)\Graphviz2.38\bin"

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
            out_file=dot_data,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True, 
            rounded=True,
            impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
