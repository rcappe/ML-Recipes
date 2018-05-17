# https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal

# Iris flower data set
# https://en.wikipedia.org/wiki/Iris_flower_data_set

from sklearn.datasets import load_iris
iris = load_iris()
print (iris.feature_names)
print (iris.target_names)
print (iris.data[0])

for i in range(len(iris.target)):
    print ("Exemple %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

