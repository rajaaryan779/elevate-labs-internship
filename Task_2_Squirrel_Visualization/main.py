### Import Necessary Libraries ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Load the data ###
df = pd.read_csv('2018_Central_Park_Squirrel_Census_-_Squirrel_Data_20250526.csv')

### See First Few Rows ###
print(df.head())

### Basic Info About Data ###
print(df.info())

### Summary of Numbers ###
print(df.describe())

### Checking if there are any Missing Values ###
print(df.isnull().sum())

### Simple Histogram (Age) ###
df['Age'].value_counts().plot(kind='bar')
plt.title('Age Distribution of Squirrels')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

### Simple Bar Plot (Fur Color) ###
df['Primary Fur Color'].value_counts().plot(kind='bar', color='orange')
plt.title('Primary Fur Color of Squirrels')
plt.xlabel('Fur Color')
plt.ylabel('Count')
plt.show()

### Simple Box Plot (Hectare Squirrel Number) ###
sns.boxplot(x='Hectare Squirrel Number', data=df)
plt.title('Squirrel Count in Hectares')
plt.show()

### Simple Correlation Heatmap ###

### Get numeric data ###
num_data = df.select_dtypes(include=['int64', 'float64'])

### Draw heatmap ###
sns.heatmap(num_data.corr(), annot=True, cmap='Blues')
plt.title('Correlation between Numeric Columns')
plt.show()

### Easy Pair Plot ###
sns.pairplot(df[['X', 'Y', 'Hectare Squirrel Number']])
plt.show()

### Pie Chart: Squirrel Shift (AM/PM) ###
df['Shift'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Squirrel Sightings by Shift')
plt.ylabel('')
plt.show()

### Count plot: Primary Fur Color by Shift ###
sns.countplot(data=df, x='Primary Fur Color', hue='Shift')
plt.title('Fur Color by Shift (AM/PM)')
plt.xticks(rotation=30)
plt.show()

### Scatter Plot: X vs. Y (Location Plotting) ###
plt.figure(figsize=(8, 6))
plt.scatter(df['X'], df['Y'], alpha=0.5, s=10)
plt.title('Squirrel Sightings in Central Park')
plt.xlabel('Longitude (X)')
plt.ylabel('Latitude (Y)')
plt.show()




