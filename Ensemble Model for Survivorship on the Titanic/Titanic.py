import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model, preprocessing, datasets, svm, metrics
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import seaborn as sb 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#load data
path = "/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Kaggle/Titanic/"
train_data = pd.read_csv(path+"train.csv"); train_data.head()
#embarked - drop missing embarked #% of missing features
(train_data.isnull().sum())*100/len(train_data.index)
#drop embarked na
train_data = train_data.dropna(subset=['Embarked'])
#create target variable
target = train_data["Survived"]; target.head()
#create features dataframe - drop passengerid, name, survived, ticket, cabin
features = train_data.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis = 1); features.head()

#impute missing values
#age
#method of dealing with missing data
"""linear regression for missing age values 
missing_age = features[features.Age.isnull()]   #ages I would like to predict 
linear_data = features.dropna(subset=['Age', 'Embarked']); linear_data.isnull().sum()

target_age = linear_data.Age
features_linear = linear_data.drop(["Age"], 1)
x_train, x_test, y_train, y_test = train_test_split(features_linear, target_age, test_size = 0.1, random_state = 1)

train_data = pd.concat([x_train, y_train], axis = 1); train_data.head()
#linear regression using data without missing values as training data
lm_model_smf = smf.ols(formula = 'Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked', data = train_data).fit()
lm_model_smf.summary()
prediction = lm_model_smf.predict(x_test)
r2_score(y_test, prediction)   #under 25 % accuracy - might have to go with median or mean

using SimpleImputer
#impute with median age and drop missing cabin
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(features[["Age"]])
features["Age"] = imputer.transform(features[["Age"]]).ravel()
features.isnull().sum()"""

age_mean = features.Age.mean()
age_median = features.Age.median()
age_std = features.Age.std()  #there is a large standard deviation, so we can opt for using the median (affected less by outliers)
#or we can opt for imputing with a random age in the range (mean +/- std)
random_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = features["Age"].isnull().sum())
age_rand = features["Age"].copy()
age_rand[np.isnan(age_rand)] = random_age
features["Age"] = age_rand
#new variables and variable edits
features["Family_Members"] = features.SibSp + features.Parch
features.Sex[features.Sex == 'male'] = 1
features.Sex[features.Sex == 'female'] = 0; features["Sex"] = features.Sex.astype('int64')

features.Embarked[features.Embarked == 'C'] = 1
features.Embarked[features.Embarked == 'Q'] = 2
features.Embarked[features.Embarked == 'S'] = 3; features["Embarked"] = features.Embarked.astype('int64')

#correlation matrix
fig, ax = plt.subplots(figsize=(7,7)) 
sb.heatmap(pd.concat([target, features], axis = 1)[["Survived", "Age", "Pclass", "Sex", "Embarked", "Family_Members"]].corr(method = "pearson"), annot = True, square=True, cmap= 'coolwarm', ax = ax)

#features - categorical variables 
features['Pclass'] = features['Pclass'].astype('category')
features['Sex'] = features['Sex'].astype('category')
features['Embarked'] = features['Embarked'].astype('category')
features = features.drop(["SibSp", "Parch"], axis = 1); features.head()

#Data Exploration
#exploratory data analysis - frequency distribution
sb.pairplot(features)
sb.distplot(features.Pclass, kde = False, bins = 5)
sb.distplot(features.Family_Members, kde = False, bins = 11)
sb.distplot(features.Embarked, kde = False, bins = 5)

#load and clean test data
#preprocess the testing data
test_data = pd.read_csv(path+"test.csv"); test_data.head()
test_data.isnull().sum()   #too many missing cabin, drop cabin

features_test = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1); features.head()
features_test.isnull().sum()  #age and fare have missing values

#impute age as with trainig data 
age_mean_test = features_test.Age.mean()
age_median_test = features_test.Age.median()
age_std_test = features_test.Age.std()  #there is a large standard deviation, so we can opt for using the median (affected less by outliers)
#or we can opt for imputing with a random age in the range (mean +/- std)
random_age_test = np.random.randint(age_mean_test - age_std_test, age_mean_test + age_std_test, size = features_test["Age"].isnull().sum())
age_rand_test = features_test["Age"].copy()
age_rand_test[np.isnan(age_rand_test)] = random_age_test
features_test["Age"] = age_rand_test

#impute missing fare with median from class 
features_test[features_test.Fare.isnull()].Pclass  #traveler belongs to class 3
features_test["Fare"] = features_test["Fare"].fillna(features_test.groupby('Pclass').mean()['Fare'][3])
features_test.isnull().sum()   #no missing values

#convert pclass, sibsp, parch and embarked to factors
features_test['Pclass'] = features_test['Pclass'].astype('category')
features_test.Sex[features_test.Sex == 'male'] = 1
features_test.Sex[features_test.Sex == 'female'] = 0
features_test['Sex'] = features_test['Sex'].astype('category')
features_test["Family_Members"] = features_test.SibSp + features_test.Parch
features_test.Embarked[features_test.Embarked == 'C'] = 1
features_test.Embarked[features_test.Embarked == 'Q'] = 2
features_test.Embarked[features_test.Embarked == 'S'] = 3
features_test['Embarked'] = features_test['Embarked'].astype('category')
features_test = features_test.drop(["SibSp", "Parch"], axis = 1); features_test.head()
features_test.isnull().sum(); features_test.shape  #preprocess complete

#Using various classifiers
#x_train and y_train
x_train = features
y_train = target
#Method 1: Logistic Regression 
logit_titanic_model = LogisticRegression(random_state = 0).fit(x_train, y_train)
logit_prediction = logit_titanic_model.predict(features_test)
#evaluating the model - cross validation score
logit_score = round(cross_val_score(logit_titanic_model, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)

#Method 2: K-Nearest Neighbors - https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
k = [1,3,5,7,9,11,13,15]
for i in k:
    titanic_knn_model = KNeighborsClassifier(n_neighbors=i)
    titanic_knn_model.fit(x_train, y_train)
    score = round(cross_val_score(titanic_knn_model, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)
    print(score)   #seems a k = 3 is the best

titanic_knn_model = KNeighborsClassifier(n_neighbors=3)
titanic_knn_model.fit(x_train, y_train)
knn_prediction = titanic_knn_model.predict(features_test)
knn_score = round(cross_val_score(titanic_knn_model, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)

#Method 3: SMVs
#add kernel parameters
titanic_svm_model = svm.SVC(kernel="linear")
titanic_svm_model.fit(x_train, y_train)
smv_predictions = titanic_svm_model.predict(features_test)
smv_score = round(cross_val_score(titanic_svm_model, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)

#Method 4 : Random Forest 
random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_train, y_train)
random_forest_predictions = random_forest_model.predict(features_test)
random_forest_score = round(cross_val_score(random_forest_model, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)

#Method 5: Decision Treet
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
decision_tree_predictions = decision_tree_model.predict(features_test)
decision_tree_score = round(cross_val_score(decision_tree_model, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)

#from the scores, our model ensemble will have smv, logit, random forest and decision tree
#Create model ensemble
logit = LogisticRegression()
smv = svm.SVC(kernel="linear")
randforest = RandomForestClassifier()
dectree = DecisionTreeClassifier()

titanic_multimodel = VotingClassifier(estimators = [('logit', logit), ('smv', smv), ('randforest', randforest), ('dectree', dectree)], voting='hard')
titanic_multimodel.fit(x_train, y_train)
multimodel_predictions = titanic_multimodel.predict(features_test)
multimodel_score = round(cross_val_score(titanic_multimodel, x_train, y_train, cv=10, scoring = "accuracy").mean()*100, 4)

#final prediction
final_prediction = titanic_multimodel.predict(features_test)
final_prediction = pd.DataFrame(final_prediction)
final_prediction.columns = ["Survived"]; final_prediction.head()

submission = pd.concat([test_data.PassengerId, final_prediction], axis = 1)
submission.to_csv (r'/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Kaggle/Titanic/submission_2.csv', index = False, header=True)