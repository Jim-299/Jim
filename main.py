# LOADING LIBRARY
import joblib
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# LOADING DATASET
dataset = read_csv("IRIS.csv")

# LOOKING AT THE DATASET
print(dataset.shape)
print(dataset.head(5))     # print first 10 rows of the dataset
print(dataset.describe())   # print summary of statistics

print(dataset.groupby('species').size())    # species distribution in the dataset

# DATA VISUALIZATION
# Uni-variate plots, better understanding each attribute
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)      # boxplot
pyplot.show()

dataset.hist()      # histogram
pyplot.show()

# Multi-variate plots, understand interaction between variables
scatter_matrix(dataset)
pyplot.show()

# EVALUATING ALGORITHMS

# Creating a validation dataset
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Building a Model
# spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# comparing algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithms Comparison')
pyplot.show()

# MAKE PREDICTIONS
# using support vector machine since it had the accuracy during training
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)


# # saving a model, this is a side quest
# filename = "First_SVC_Model.sav"
# joblib.dump(model, filename)
#
# #  loading and testing a model
# loaded_model = joblib.load(filename)
# result = loaded_model.score(x_validation, y_validation)
# print(result)  #

# evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))









