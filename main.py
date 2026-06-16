# importing libraries
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# To Load dataset
df = pd.read_csv(r"C:\Users\phani\OneDrive\Desktop\Titanic-Survival-Prediction\Titanic_Dataset.csv")

# Check for missing values
print(df.isnull().sum())

# TO handle missing values in Age column
df['Age'] = df['Age'].fillna(df['Age'].median())

# TO handle missing values in Embarked column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop the Cabin,Name,Ticket columns as they have a large number of missing values and are not likely to be useful for prediction
df.drop('Cabin', axis=1, inplace=True)
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Check for missing values again after handling
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model building
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)
pred_labels = ["Survived" if x == 1 else "Not Survived" for x in y_pred]

# Print the predicted labels for the first 10 test samples actual vs predicted
print(pred_labels[:10])
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
results["Actual"] = results["Actual"].map({1: "Survived", 0: "Not Survived"})
results["Predicted"] = results["Predicted"].map({1: "Survived", 0: "Not Survived"})
print(results.head(10)) 

# model accuracy and confusion matrix
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))