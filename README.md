# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. import pandas module and import the required data set.

2. Find the null values and count them.

3. Count number of left values.

4. From sklearn import LabelEncoder to convert string values to numerical values.

5. From sklearn.model_selection import train_test_split.

6. Assign the train dataset and test dataset.

7. From sklearn.tree import DecisionTreeClassifier.

8. Use criteria as entropy.

9. From sklearn import metrics.

10. Find the accuracy of our model and predict the require values. 

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Ravikrishnamoorthy D

RegisterNumber: 212224040271

*/

```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```

## Output:

![image](https://github.com/user-attachments/assets/54d4a5ca-9f31-4b6b-bdba-4ba44e2d2603)

![image](https://github.com/user-attachments/assets/6b80abf3-ede6-457f-84c6-f04bcf8e1143)

![image](https://github.com/user-attachments/assets/f857b009-9531-449a-a5bb-7ccdf22e664f)

![image](https://github.com/user-attachments/assets/9bb5d70a-2951-4af5-a1f9-890e3a89cf1f)

![image](https://github.com/user-attachments/assets/e182fa72-3f9e-409d-9967-5c7b59e11b02)

![image](https://github.com/user-attachments/assets/aabdd293-335f-44ee-a497-9856b21e2f11)

![image](https://github.com/user-attachments/assets/c8fea09b-015b-4a97-bc66-d66048a03241)

![image](https://github.com/user-attachments/assets/73672786-2f52-4965-b38e-c6975a825970)








## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
