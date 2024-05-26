import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from pandas.plotting import scatter_matrix
from matplotlib import cm


fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()
print(fruits.shape) #printing the types of fruits and adding to a graph
print(fruits['fruit_name'].unique())
print(fruits.groupby('fruit_name').size())
'''
sns.countplot(fruits['fruit_name'], label="Count")
plt.show()

fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False, figsize=(9,9), title='Boxplot for each variable')
plt.savefig('fruit_boxplot')
plt.show()

fruits.drop('fruit_label' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_hist')
plt.show()
'''

# Assuming fruits DataFrame is already defined
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

# Create scatter matrix
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9))

# Add title and save plot
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix.png')
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
logreg = LinearRegression()
logreg.fit(X_train, y_train)
print('Accuracy of logistic regression model on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of linear regression model on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
