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
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# print(dataset.shape,type(dataset))
# print(dataset.head(20))
# print(dataset.describe())
# print(dataset.groupby('class').size())
# dataset.plot(kind='box',subplots=True, layout=(2,2),sharex=False, sharey=False)
# pyplot.show()
# dataset.hist()
# pyplot.show()
# scatter_matrix(dataset)
# pyplot.show()

arr = dataset.values
X = arr[:,0:4]
y = arr[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=.2, random_state=1)
# print(X, y)
# print(X_train, X_validation, Y_train, Y_validation)

#Spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

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

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
