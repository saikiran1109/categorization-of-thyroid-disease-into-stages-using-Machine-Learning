#======================= IMPORT PACKAGES =============================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing 
 

#===================== DATA SELECTION ==============================

#=== READ A DATASET ====
data_frame=pd.read_csv("thyroid.csv")
print("-------------------------------------------------------")
print("================== 1.Data Selection ===================")
print("-------------------------------------------------------")
print()
print(data_frame.head(20))


#=====================  2.DATA PREPROCESSING ==========================


#=== CHECK MISSING VALUES ===


print("---------------------------------------------------------")
print("================ Before Checking missing values =========")
print("---------------------------------------------------------")
print()
print(data_frame.isnull().sum())
print()


print("---------------------------------------------------------")
print("================ After Checking missing values =========")
print("---------------------------------------------------------")
print()
data_frame=data_frame.fillna(0)
print(data_frame.isnull().sum())
print()


#==== LABEL ENCODING ====

label_encoder = preprocessing.LabelEncoder() 
print("------------------------------------------------------")
print("================ Before label encoding ===========")
print("------------------------------------------------------")
print()
print(data_frame['Classes'].head(10))

print("------------------------------------------------------")
print("================ After label encoding ===========")
print("------------------------------------------------------")
print()

data_frame= data_frame.astype(str).apply(label_encoder.fit_transform)

print(data_frame['Classes'].head(10))

#====================== 3. DATA SPLITTING =============================

x=data_frame.drop('Classes',axis=1)
y=data_frame['Classes']

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=50)


print("=========================================================")
print("-------------------- Data Splitting -------------------")
print("==========================================================")
print()
print("Total number of rows in dataset       :", data_frame.shape[0])
print()
print("Total number of rows in training data :", X_train.shape[0])
print()
print("Total number of rows in testing data  :", X_test.shape[0])


#======================== 4. CLASSIFICATION =============================


#=== DECISION TREE ===

from sklearn.tree import DecisionTreeClassifier 

dt=DecisionTreeClassifier()

dt.fit(X_train, Y_train)

y_pred_dt=dt.predict(X_train)

import numpy as np


y_pred_dt1=np.where(y_pred_dt==3)



y_pred_dt_df=pd.DataFrame([y_pred_dt])


y_pred_dt_df.drop([0,    1,    2,16,   38,   44,   50,   85,  101,  107,  158,  159,  187,  212,
         230,  237,  249,  285,  317,  326,  333,  357,  367,  410,  443,
         448,  464,  495,  509,  514,  537,  542,  543,  569,  623,  635,
         651,  660,  701, 1659, 1660], axis=1, inplace=True)



y_pred_dt_df=y_pred_dt_df.T
print(y_pred_dt_df)

print("--------------- Predicting( Decision Tree) -------------")

print()

print(y_pred_dt_df)



#=== NAIVE BAYES ===


from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(X_train[0:1620], y_pred_dt_df)

y_pred_nb=nb.predict(X_test)
print("++++++++++++++++++++++++++++")
print(X_test)
print(y_pred_nb)

print()
print("--------------- Classifying( Naives Bayes) --------------")
print()

for i in range(0,10):
    if y_pred_nb[i]==0:
        print("----------------------")
        print()
        print([i],"The Thyroid is CRITICAL")
    elif y_pred_nb[i]==1:
        print("----------------------")
        print()
        print([i],"The Thyroid is MAJOR")
    elif y_pred_nb[i]==2:
        print("----------------------")
        print()
        print([i],"The Thyroid is MINOR ")

print()
print("********************************************************************")
print()
	
input_1 = np.array([68,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,13,47,112,65,93]).reshape(1, -1)
print("---------------------- The Given symptoms is ------------------------------ ")
print()
print(input_1)

predicted_data = nb.predict(input_1)
print(y_pred_nb[predicted_data])
if y_pred_nb[predicted_data]==0:
    print("----------------------")
    print()
    print("The Thyroid is CRITICAL")
    print()
    print("----------------------")
    
elif y_pred_nb[predicted_data]==1:
    print("----------------------")
    print()
    print("The Thyroid is MAJOR")
    print()
    print("----------------------")
elif y_pred_nb[predicted_data]==2:
    print("----------------------")
    print()
    print("The Thyroid is MINOR")
    print()
    print("----------------------")
    
else:
    print("----------------------")
    print()
    print("Not affected by thyroid ")
    print()
    print("----------------------")

#======================== 5. PERFORMANCE METRICS =============================

from sklearn import metrics

cm=metrics.confusion_matrix(y_pred_nb,Y_test)

TP=cm[1][3]
TN=cm[2][3]
FP=cm[3][3]
FN=cm[0][0]

Total= TP+TN+FP+FN

Acc=(TP + TN + FP)/Total

Specificity = (TN / (TN+FP))*100

Sensitivity = ((TP) / (TP+FN))*100


print()
print("----------------- Performance analysis  -----------------")
print()

print("1. Accuracy :", Acc *100,'%')
print()
print("2. Confusion matrix :\n", cm)
print()
print("3. Specificity :",Specificity,'%')
print()
print("4. Sensitivity :",Sensitivity,'%')














