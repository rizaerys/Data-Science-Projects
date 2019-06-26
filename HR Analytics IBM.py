# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:07:56 2019

@author: User
"""
#Importing the required libraries and dependancies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, precision_score, accuracy_score
from sklearn.metrics import recall_score, classification_report, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

#Loading the dataset
hr = pd.read_csv("HR Analytics - Attrition.csv")

#Exploring the data set
hr.head(10)
hr.shape
hr.size
hr.columns
len(hr)
len(hr.columns)
hr.dtypes
hr.describe(include='all')
hr.corr()


#Checking for duplicate rows and dropping if any
sum(hr.duplicated())
hr.drop_duplicates(keep= 'first',inplace=False) 
#Retains the first column and drops the remaining
sum(hr.duplicated())


#Checking for missing data
hr.isnull().sum()
hr.isnull().count()

#Checking the count and different data types in DF
for column in hr:
    print(hr[column].astype('category').value_counts())
    print('___________________________________________________')

#dropping columns having single value - EmployeeCount, StandardHours, Over18 
hr.drop(['EmployeeCount','Over18','StandardHours'], axis = 1, inplace = True)
hr.head(10)
hr.shape
hr.size

#Correlation Matrix and correlation heat map
corr = hr.corr()
corr
plt.figure(figsize=(20,15))
sns.heatmap(corr, annot=True)
plt.show()

# Plotting of various factors
sns.countplot(x = "EducationField", hue = "Gender", data = hr)
sns.countplot(x = "Department", hue = "Gender", data = hr)
sns.catplot(x="EnvironmentSatisfaction", y="Attrition", kind="box", data=hr)
sns.catplot(x="Business_Travel", y="Attrition", kind="box", data=hr)
sns.catplot(x="Age", y="MonthlyIncome",bins = 10, kind ="violin", data=hr)
sns.catplot(y="Attrition", x="MonthlyIncome",bins = 10, kind ="violin", data=hr)
sns.countplot(x = "Department",)
x = hr["MonthlyIncome"]
sns.distplot(x,bins = 10, kde_kws={"color": "k", "lw": 3, "label": "KDE"}, rug=True, color = "y")

#Outlier Check  - IQR Generation 
# Outlier treatment for Age 
Q1 = hr.Age.quantile(0.25)
Q3 = hr.Age.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.Age >= Q1 - 1.5*IQR) & (hr.Age <= Q3 + 1.5*IQR)]
# Outlier treatment for DailyRate 
Q1 = hr.DailyRate.quantile(0.25)
Q3 = hr.DailyRate.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.DailyRate >= Q1 - 1.5*IQR) & (hr.DailyRate <= Q3 + 1.5*IQR)]
# Outlier treatment for HourlyRate 
Q1 = hr.HourlyRate.quantile(0.25)
Q3 = hr.HourlyRate.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.HourlyRate >= Q1 - 1.5*IQR) & (hr.HourlyRate <= Q3 + 1.5*IQR)]
# Outlier treatment for TotalWorkingYears
Q1 = hr.TotalWorkingYears.quantile(0.25)
Q3 = hr.TotalWorkingYears.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.TotalWorkingYears >= Q1 - 1.5*IQR) & (hr.TotalWorkingYears <= Q3 + 1.5*IQR)]
# Outlier treatment for YearsAtCompany 
Q1 = hr.YearsAtCompany.quantile(0.25)
Q3 = hr.YearsAtCompany.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.YearsAtCompany >= Q1 - 1.5*IQR) & (hr.YearsAtCompany <= Q3 + 1.5*IQR)]
# Outlier treatment for YearsInCurrentRole 
Q1 = hr.YearsInCurrentRole.quantile(0.25)
Q3 = hr.YearsInCurrentRole.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.YearsInCurrentRole >= Q1 - 1.5*IQR) & (hr.YearsInCurrentRole <= Q3 + 1.5*IQR)]
# Outlier treatment for YearsSinceLastPromotion 
Q1 = hr.YearsSinceLastPromotion.quantile(0.25)
Q3 = hr.YearsSinceLastPromotion.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.YearsSinceLastPromotion >= Q1 - 1.5*IQR) & (hr.YearsSinceLastPromotion <= Q3 + 1.5*IQR)]
# Outlier treatment for YearsWithCurrManager 
Q1 = hr.YearsWithCurrManager.quantile(0.25)
Q3 = hr.YearsWithCurrManager.quantile(0.75)
IQR = Q3 - Q1
hr = hr[(hr.YearsWithCurrManager >= Q1 - 1.5*IQR) & (hr.YearsWithCurrManager <= Q3 + 1.5*IQR)]
hr
hr.shape


#Mapping binary variables to 1 and 0 and defining map function
varlist =  ['OverTime','Attrition']

def binary_map(x):
    return x.map({'Yes': 1, 'No': 0})

# Applying the function to the leads score list
hr[varlist] = hr[varlist].apply(binary_map)# List of binary variables with Yes/No values using map converting these to 1/0
varlist =  ['OverTime','Attrition']

# Dummy variable encoding and concatenation to the DF
dummy = pd.get_dummies(hr[['Gender','JobRole', 'EducationField', 'Department','MaritalStatus','BusinessTravel']], drop_first=True)
hr = pd.concat([hr, dummy], axis=1)

# Dropping the repeated variables as we have created dummies for the below variables
hr = hr.drop(['Gender','JobRole', 'EducationField', 'Department','MaritalStatus','BusinessTravel'], 1)
print(hr.shape)
hr.head()



#Atrition Count
hr['Attrition'].value_counts()

#Specifying dependant and independant variable
X = hr.drop('Attrition', axis =1)
X
Y = hr['Attrition']
Y



# Splitting the data into train and test and Scaling it 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)


vif = [variance_inflation_factor(Xtrain, i) for i in range(Xtrain.shape[1])]
vif

#Attrition %
AttritionRate = round((sum(hr['Attrition'])/len(hr['Attrition'].index))*100)
AttritionRate


#Modelling using Logistic Regression
logreg = LogisticRegression()
logreg.fit(Xtrain,Ytrain)

#Prediction
Ypred = logreg.predict(Xtest)

#Logistic Regression Metrics
cm = confusion_matrix(Ytest,Ypred)
cm
acuscore = accuracy_score(Ytest, Ypred)
JSI = jaccard_similarity_score(Ytest, Ypred)
print("Accuracy Score in Logistic Regression :", acuscore*100, "%")
print("Jaccard Similarity Index : " , JSI*100, "%")


TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives

False_Positive_Rate  = (FP/ float(TN+FP))
Positive_Predictive_value = (TP / float(TP+FP))
Negative_Predictive_value = (TN / float(TN+ FN))
Sensitivity = TP / float(TP+FN)
Selectivity = TN / float(TN+FP)
print("Sensitivity is:", Sensitivity)
print("Selectivity is:", Selectivity)
print("Negative_Predictive_value : ",Negative_Predictive_value)
print("Positive_Predictive_value : ", Positive_Predictive_value)
print("False_Positive_Rate : ", False_Positive_Rate)
print(classification_report(Ytest,Ypred))


#Decision Tree Classification
DT = DecisionTreeClassifier(random_state=0, max_depth=2)
DT = DT.fit(Xtrain, Ytrain)
DTPred = DT.predict(Xtest)
AcuScDT = accuracy_score(Ytest,DTPred)
print("Accuracy in Decision Tree Classifier:", AcuScDT*100 ,"%") 
print(classification_report(Ytest,DTPred))

#KNN classifier
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(Xtrain, Ytrain)
KNNpred = knn.predict(Xtest)
KNNpred
AcuScKNNC = accuracy_score(Ytest,KNNpred)
print("Accuracy in KNN Classifier:", AcuScKNNC*100 ,"%") 
print(classification_report(Ytest,KNNpred))


#Evaluation
print("Accuracy Score in Logistic Regression :", acuscore*100, "%")
print(classification_report(Ytest,Ypred))
print("Accuracy in Decision Tree Classifier:", AcuScDT*100 ,"%") 
print(classification_report(Ytest,DTPred))
print("Accuracy in KNN Classifier:", AcuScKNNC*100 ,"%") 
print(classification_report(Ytest,KNNpred))

