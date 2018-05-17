from sklearn import tree
#import sklearn

# Step 1
# we set up the features and labeles
# features = [[140, "smooth"], [130,"smooth"], [150, "bumpy"],[170, "bumpy"]]
# labeles = ["apple","apple","orange","orange"]

# Step 2
# we replace the string with numbers
features = [[140, 1], [130,1], [150, 0],[170, 0]]
labeles = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labeles)


print (clf.predict([[130,1]])) #orange
print (clf.predict([[160,0]])) #apple