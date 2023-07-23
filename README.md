# Mental-Fitness-Tacker

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn

from google.colab import drive
drive.mount('/content/drive')

df1=pd.read_csv("/content/drive/MyDrive/Data Sets/mental-and-substance-use-as-share-of-disease.csv")
df2=pd.read_csv("/content/drive/MyDrive/Data Sets/prevalence-by-mental-and-substance-use-disorder.csv")

df1.head()

df2.head()

df1.describe(),df1.info()
df2.describe(),df2.info()

df=pd.concat(objs=[df2,df1],axis=1)

corr=df.corr()
plt.figure(figsize=(15,12))
sns.pairplot(df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',]])

 plt.figure(figsize=(15,12))
sns.heatmap(corr)

df.drop(['Entity','Code','Year'],axis=1,inplace=True)
df=df.fillna(df.mean())

x=df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',]].to_numpy()

y=df[['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']].to_numpy()

scaler=StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y)

ml=RandomForestRegressor()
ml.fit(x_train,y_train)
predicted_values=ml.predict(x_test)

plt.figure(figsize=(15,12))
plt.plot(y_test[:100])
plt.plot(predicted_values[:100])
plt.legend(['true','predicted'])
plt.title('Mean Square Error '+str(sklearn.metrics.mean_squared_error(y_test,predicted_values)))
plt.show()

