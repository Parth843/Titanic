import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

# Function cleans the input data frame to make it suitable for analysis.
def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
    
    data['Embarked'] = data['Embarked'].fillna("S")
    
# Import training and validation data and apply the clean_data function on it.
train = pd.read_csv('train.csv')

clean_data(train)
target = train['Survived'].values
train.drop(['Survived'], axis = 1,inplace = True)
train.set_index('PassengerId', inplace = True)

# Select the relevant features from the dataset.
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
X = pd.get_dummies(X) # One hot encoding for  the categorical features

# Import and clean the test data.
test = pd.read_csv('test.csv')
clean_data(test)
x_test = test[features]
x_test = pd.get_dummies(x_test) # One hot encoding for the categorical features

# Split the data into train and validation sets.
xtrain, xval, ytrain, yval = train_test_split(X,target,test_size = 0.25,random_state = 1,stratify = target)

# Find the best support vector classifier for the data
pipe_svc = make_pipeline(StandardScaler(),PCA(n_components = 7),SVC(random_state = 1))
param_range = [0.0001, 0.001 ,0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__kernel':['rbf'],'svc__gamma':param_range}]
gs = GridSearchCV(estimator = pipe_svc,param_grid = param_grid,scoring = 'accuracy',cv = 10,refit = True,n_jobs = -1)
gs.fit(xtrain, ytrain)
print(gs.best_score_)
print(gs.best_params_)

# Test accuracy of the best support vector classifier on validation set
pipe_svc1 = make_pipeline(StandardScaler(),PCA(n_components = 7),SVC(kernel = 'rbf',C = 1,gamma = 0.1))
pipe_svc1.fit(xtrain, ytrain)
yhat_svm = pipe_svc1.predict(xval)
np.mean(yval.reshape(1, -1) == yhat_svm)
# 0.8295964

# Find best logistic Regression classifier.
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components = 7), LogisticRegression())
param_range = [0.0001, 0.001 ,0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = {'logisticregression__C':param_range}
gs1 = GridSearchCV(estimator = pipe_lr,param_grid = param_grid,scoring = 'accuracy',cv = 10,refit = True, n_jobs = -1)
gs1.fit(xtrain, ytrain)
print(gs1.best_score_)
print(gs1.best_params_)

# Test accuracy of the best logistic classifier on validation data. 
pipe_lr1 = make_pipeline(StandardScaler(),PCA(n_components = 5),LogisticRegression(C = 0.01))
pipe_lr1.fit(xtrain, ytrain)
yhat_lr = pipe_lr1.predict(xtest)
np.mean(yhat_lr == yval.reshape(1,-1))
# 0.79372

# Make a random forest calssifier.
rfc = RandomForestClassifier(criterion = 'entropy')
rfc.fit(xtrain, ytrain)
yhat_tree = rfc.predict(xval)
# Test random forest accuracy for validation data.
np.mean(yhat_tree == yval.reshape(1,-1))
# 0.825112

# Use majority voting on validation data to make final predictions.
yval = yval.reshape(1,-1)[0]
score = []
counter = 0

for x in zip(yhat_tree,yhat_svm,yhat_lr):
    prediction = stats.mode(x)[0][0]
    if yval[counter] == prediction:
        score.append(1)
    else:
        score.append(0)
    counter += 1
        
sum(score)/len(score)
# 0.82511210

# Make predictions for the test dataset by individural calssifiers.
predictions_svm = pipe_svc1.predict(x_test)
predictions_lr = pipe_lr1.predict(x_test)
predictions_rfc = rfc.predict(x_test)

# Make final predictioins by using majority voting.
final_preds = []
for x in zip(predictions_svm, predictions_lr, predictions_rfc):
    prediction = stats.mode(x)[0][0]
    final_preds.append(prediction)
    
# Save the result.
df = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})
df.to_csv('titanic_rfc.csv',index = False)   

# 0.77336 score on kaggle