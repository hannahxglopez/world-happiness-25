"""
Course Project

hlopez24@georgefox.edu
"""
# import stuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
import sklearn.model_selection
import sklearn.svm
import sklearn.metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# from scipy.special import y_pred

le = LabelEncoder()

# load data set
data = pd.read_csv("world_happiness_combined.csv", delimiter=';', decimal=',')
data['target'] = pd.cut(data['Ranking'], bins=4, labels=[0, 1, 2, 3], ordered=True)
class_labels = {0: "Most Happy", 1: "Happy-ish", 2: "Unhappy-ish", 3: "Most Unhappy"}
data['target_name'] = data['target'].map(class_labels)

# encode and discretize columns to use as X and y
data['Regional indicator numbers'] = le.fit_transform(data['Regional indicator'])
# check data
# Healthy life expectancy is varied (good for X)
# Generosity isn't very diverse for at least this value (bad for X)
# GDP per capita is different for each country but the value counts (the amount in each category)
# is almost the same in each one
#smoking = data[data['target'] == 4]
# smoking['Regional indicator'].value_counts().plot.bar()
#plt.show()
# exit()

# separate into parallel X and y arrays and stratify data
X = data[['Healthy life expectancy', 'Social support', 'Freedom to make life choices', 'Regional indicator numbers', 'GDP per capita', 'Generosity']].to_numpy()
y = data[['target']].to_numpy().reshape((-1,))

# create 3 figures to visually describe three different aspects of your data
#  save these as pdf files (format: fig#.pdf)
#  one of these must plot features from X
plt.figure()
plt.scatter(data['Social support'], data['Freedom to make life choices'], c=data['target'], cmap='coolwarm', alpha=0.6)
plt.xlabel('Social support')
plt.ylabel('Freedom to make life choices')
plt.title('Social Support vs Freedom to make life choices (colored by Happiness Ranking)')
plt.colorbar(label='Ranking (0 = Most Happy, 3 = Most Unhappy)')
plt.savefig('fig1.pdf')
plt.show()

crosstab = pd.crosstab(data['target_name'], data['Regional indicator'])
crosstab.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Happiness Category')
plt.ylabel('Number of Countries')
plt.title('Regional Distribution within Happiness Categories')
plt.savefig('fig2.pdf')
plt.show()

data.boxplot(column='Regional indicator numbers', by='target')
plt.title('Regional Indicator by Happiness Ranking')
plt.suptitle('')
plt.xlabel('Happiness Ranking')
plt.ylabel('Regional Indicator')
plt.savefig('fig3.pdf')
plt.show()


# use appropriate cross-validation technique to train and test an SVM-based
#  classification model (test set: 10-20% of data)
k_fold_helper = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True)
fold = 1
y_test_all = None
y_pred_all = None
for train_indices, test_indices in k_fold_helper.split(X, y):
	print(f'n*** fold {fold}', '*' * 40)

	X_train = X[train_indices]
	y_train = y[train_indices]
	X_test = X[test_indices]
	y_test = y[test_indices]

	# standardize features to remove the mean and scale according to the standard
	# deviation. This helps increase accuracy (normally).
	scaler = preprocessing.StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	model = sklearn.svm.SVC(kernel='rbf')
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	# compute metrics for the current fold only using y_test (y_true) and y_pred
	# print('true:', y_test)
	# print('pred:', y_pred)

	accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
	print(f'accuracy: {accuracy:.3}')

	balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
	print(f'balanced accuracy: {balanced_accuracy:.3}')

	# confusion matrix!
	cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
	print('conufusion matrix:')
	print(cm)

	# now, go keep track of the current fold's y_test and y_pred by concatenating them
	# onto the end of the previous results

	# if we don't have previous results yet, y_test_all is still None
	# so, make the current y_test be the new y_test_all

	if y_test_all is None:
		y_test_all = np.copy(y_test)
		y_pred_all = np.copy(y_pred)
	else:
		# else, we do have previous results, so simply concatenate/"horizontally stack" (hstack)
		# the current 15 onto the end of the previous stuff
		y_test_all = np.hstack((y_test_all, y_test))
		y_pred_all = np.hstack((y_pred_all, y_pred))

	fold += 1


# compute the raw and balanced accuracy for each cross-validation fold,
#  as well as the overall raw and balanced accuracy
print(f'\n*** all folds', '*' * 40)

print('true:', y_test_all)
print('pred:', y_pred_all)

accuracy = sklearn.metrics.accuracy_score(y_test_all, y_pred_all)
print(f'accuracy: {accuracy:.3f}')

balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test_all, y_pred_all)
print(f'balanced accuracy: {balanced_accuracy:.3}')

cm = sklearn.metrics.confusion_matrix(y_test_all, y_pred_all)
print('confusion matrix:')
print(cm)


# compute and plot a confusion matrix for all sammples to visually depict the
#  confusability between classes. Save this figure in a separate PDF file (cm.pdf)
#  and add to git repository
sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
    y_test_all, y_pred_all,
    cmap='Blues', colorbar=False,
    display_labels=[class_labels[i] for i in range(4)]
)
plt.savefig('cm.pdf')
plt.show()

exit()