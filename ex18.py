import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\kuldeep prajapati\OneDrive\Desktop\MLOps\Student_Data.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

X = df[['Hours']]
y = df['Marks']

print(X.shape)
print(y.shape)

np.random.seed(34)
indices = np.random.permutation(len(df))

train_size = int(0.80 * len(df))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

x_mean = X_train['Hours'].mean()
y_mean = y_train.mean()

numerator = ((X_train['Hours'] - x_mean) * (y_train - y_mean)).sum()
denominator = ((X_train['Hours'] - x_mean) ** 2).sum()

m = numerator / denominator
c = y_mean - m * x_mean

y_pred = m * X_test['Hours'] + c

print("Predicted Marks:")
print(y_pred)

mse = ((y_test - y_pred) ** 2).mean()
rmse = np.sqrt(mse)

print("Slope:", m)
print("Intercept:", c)
print("RMSE:", rmse)