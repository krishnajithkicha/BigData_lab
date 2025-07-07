import pandas as pd
import numpy as np

data={
    'Name':['Aliya','Biya','Christy','Alice',None],
    'Age':[30,31,np.nan,32,40],
    'Gender': ['Female', 'Female', 'male', 'Female', 'FEMALE'],
    'Salary': [50000, 54000, 58000, 50000, None],
    'JoinDate': ['2022-01-15', '2021-06-10', '2023-05-02', '2022-01-15', '2020-03-25'],
    'Remarks': [' Good ', 'bad', 'BAD', 'Excellent', 'good']
}

df=pd.DataFrame(data)
print("The data are:",df)

#Different Cleaning Techniques
#1.Handling Missing values

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Name'] = df['Name'].fillna('Unknown')


#2.Removing Duplicates
df.drop_duplicates(inplace=True)

#3.Converting Data Types
# Convert JoinDate to datetime, coerce errors
df['JoinDate']=pd.to_datetime(df['JoinDate'],errors='coerce')

#4.Standardizing Categorical Data
# Standardize Gender column
df['Gender']=df['Gender'].str.capitalize()
# Strip and lowercase Remarks
df['Remarks']=df['Remarks'].str.strip().str.lower()

#5.Handling Outliers
# IQR on Salaries
Q1=df['Salary'].quantile(0.25)
Q3=df['Salary'].quantile(0.75)
IQR=Q3-Q1
df=df[(df['Salary']>=Q1-1.5*IQR)&(df['Salary']<=Q3+1.5*IQR)]

print("\nCleaned DataFrame:\n", df)