import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train dataset
train_file_path = 'titan/train.csv'
train_data = pd.read_csv(train_file_path)

# Data Cleaning
# Imputing missing values for 'Age' with the median value
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Dropping the 'Cabin' column due to excessive missing values
train_data.drop(columns='Cabin', inplace=True)

# Imputing missing values for 'Embarked' with the most frequent value (mode)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Exploratory Data Analysis

# 1. Survival Rate
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=train_data, palette='Set2')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 2. Gender and Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=train_data, palette='Set1')
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 3. Class and Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=train_data, palette='Set3')
plt.title('Survival Count by Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 4. Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(train_data['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
