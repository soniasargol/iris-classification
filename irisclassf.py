# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import (train_test_split,
                                    cross_val_score,
                                    StratifiedKFold)
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("You loaded the iris dataset. This is how it looks like:")
print("######################################################")
print("------------------------------------")
print("\ndataset shape:\n",dataset.shape)
print("------------------------------------")
print("\nFirst 5 rows of the dataset:\n",dataset.head())
print("------------------------------------")
print("\nDescription of the dataset:\n",dataset.describe())
print("------------------------------------")
print("\nDataset groups and the size of each group:\n",dataset.groupby('class').size())
do_plot= input("\nDo you want to plot the dataset?[y/n]\n")
if (do_plot.lower() == 'y'):
	dataset.plot(kind='box',subplots=True, layout=(2,2),sharex=False, sharey=False,title="Box Plot")
	pyplot.show()
	dataset.hist()
	pyplot.suptitle("Histogram")
	pyplot.show()
	scatter_matrix(dataset)
	pyplot.suptitle("Scatter Matrix")
	pyplot.show()

print("------------------------------------")
print("Next step is forming the training set and test set from the data. Test size is considered to be 20% of the total size of the data.")
arr = dataset.values
X = arr[:,0:4]
y = arr[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=.2, random_state=1)
# print(X, y)
# print(X_train, X_validation, Y_train, Y_validation)

print("After forming the training and test set, we apply different machine learning models to see which one is more accurate to represent our data.")
print("------------------------------------")
print("The models used in this practice are:")
print("Logistic Regression,\nLinearDiscriminantAnalysis,\nKNeighborsClassifier,\nDecisionTreeClassifier,\nGaussianNB,\nSVM")
#Spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

print("------------------------------------")
print("Accuracy of the models are as follows:")
# Evaluate Each Model In Turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

print("------------------------------------")
print("Since SVM is the most accurate in the models, we fit that model to our training set, and use the model to make predictions on test set. The accuracy of our model is:\n")
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print("Classification report:")
print(classification_report(Y_validation, predictions))
