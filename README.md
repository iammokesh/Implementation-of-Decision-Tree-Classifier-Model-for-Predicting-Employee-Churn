# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score
## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Mokesh C 
RegisterNumber: 212225240088 
*/
import pandas as pd

data = pd.read_csv(r"C:\Users\HP\Downloads\Employee.csv")

print("data.head():")
print(data.head())

print("data.info():")
print(data.info())

print("isnull() and sum():")
print(data.isnull().sum())

print("data value counts():")
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

print("data.head() for Salary:")
data["salary"] = le.fit_transform(data["salary"])
print(data.head())

print("x.head():")

x = data[["satisfaction_level", "last_evaluation", "number_project",
          "average_montly_hours", "time_spend_company",
          "Work_accident", "promotion_last_5years", "salary"]]

print(x.head())

y = data["left"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

print("Accuracy value:")

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)

print("Data Prediction:")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plot_tree(dt,
          feature_names=x.columns,
          class_names=['Stayed', 'Left'],
          filled=True)

plt.show()
```

## Output:
<img width="900" height="504" alt="Screenshot 2026-05-16 153632" src="https://github.com/user-attachments/assets/c3fd5c20-1692-4667-a4f7-a3174b18417c" />
<img width="583" height="420" alt="Screenshot 2026-05-16 153655" src="https://github.com/user-attachments/assets/e564382c-6516-4d9f-90b0-6f6eb4219343" />
<img width="420" height="295" alt="Screenshot 2026-05-16 153739" src="https://github.com/user-attachments/assets/8db70fdb-915a-4c72-b606-b4d5655e14ce" />
<img width="378" height="118" alt="Screenshot 2026-05-16 153809" src="https://github.com/user-attachments/assets/92bb30f7-a0b8-450a-9ba7-dcd05edb9efb" />
<img width="967" height="484" alt="Screenshot 2026-05-16 153823" src="https://github.com/user-attachments/assets/808d12b9-d2b9-4ea1-94e0-5fdc7d84d9d6" />
<img width="867" height="423" alt="Screenshot 2026-05-16 153846" src="https://github.com/user-attachments/assets/112efe9f-0933-46c5-bcb3-314c26d9729b" />
<img width="1325" height="170" alt="Screenshot 2026-05-16 153555" src="https://github.com/user-attachments/assets/d89d3c98-eacc-4d6c-a990-68b958169abe" />
<img width="993" height="668" alt="Screenshot 2026-05-16 153528" src="https://github.com/user-attachments/assets/c2cf762b-b055-45e9-ab83-ebd418d7b28b" />











## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
