Step 1: Import important libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

step 2: Get data File using panda
df = pd.read_csv('/Users/rabbiyayounas/Documents/DatasetsML/insurance.csv')

step 3: check data to figure out X and Y 
df.head()

Step 4:converrt into numeric category
df['sex']=df['sex'].astype('category')
df['sex']=df['sex'].cat.codes

df['region']=df['region'].astype('category')
df['region']=df['region'].cat.codes

df['smoker']=df['smoker'].astype('category')
df['smoker']=df['smoker'].cat.codes

Step 5: to see if there are any null values to fix it 
df.isnull().sum()

Step 6: declare X and Y 
X = df.drop(columns= 'charges')
y = df['charges']

step 7 :Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


Step 8: Create polynomial features (e.g., degree 4)
poly = PolynomialFeatures(degree=4)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)


step 9: Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_poly_train, y_train)

Step 10 : doing prediction getting prediction values
y_pred = regressor.predict(X_test)
y_pred

step 11: Calculating the Coefficients
#The coefficients (regressor.coef_) indicate the impact of each feature on the predicted outcome (y).
print(regressor.coef_)

Step 12: Calculating the Intercept
#The intercept (regressor.intercept_) in this context represents the base price of a house when its size (X) is zero.
print(regressor.intercept_)

Step 13 : Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

step 14: plot to see efficientcy of data 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()
![image](https://github.com/RabbiyaYounas/Polynomial_regression-on-isnurance-data/assets/171420965/097b3b65-0757-4c95-bf0e-1d448d02d8b0)
