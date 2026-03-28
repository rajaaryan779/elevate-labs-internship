### IMPORTING REQUIRED LIBRARIES ###
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

### READING CSV FILE USING PANDAS ###
data = pandas.read_csv("2018_Central_Park_Squirrel_Census_-_Squirrel_Data_20250526.csv")

### TOP 5 ROWS ###
print(data.head())

### CHECKING IF THE SQUIRREL ID IS UNIQUE ###
print(data["Unique Squirrel ID"].is_unique)

### SETTING SQUIRREL ID COLUMN AS INDEX COLUMN ###
data.set_index("Unique Squirrel ID", inplace=True)
data.sort_index(ascending=True)
print(data)
print(data.columns)


### DIMENSIONS OF THE DATABASE ###
df = pandas.DataFrame(data)
print(df.shape)

### BASIC INFORMATION OF DATABASE ###
print(data.info())
print(data.describe())
print(data.sample(2))

### CHECKING IF THERE ARE ANY NULL VALUES ###
print(data.isna().sum())

### CHECKING IF THERE ARE ANY DUPLICATE VALUES ###
print(data.duplicated().sum())

### CREATING NEW COLUMN ###
data["Family"] = data["X"] + data["Y"]

### PRIMARY FUR COLOR OF FAMILY ###
print(sns.countplot(x="Family",hue="Primary Fur Color" , data=data))
plt.title("Primary Fur Color of Family")
print(data["Primary Fur Color"].value_counts())

### DISTRIBUTION OF Primary Fur Color ###
sns.histplot(x="Primary Fur Color",data=data,kde=True,color=sns.color_palette("muted")[1])
print(plt.title("Distribution of Primary Fur Color"))

### SCRAPING DATA FROM THE FILE AND ADDING IT TO DICTIONARY ###
grey_squirrel = len(data[data["Primary Fur Color"] == "Gray"])
cinnamon_squirrel = len(data[data["Primary Fur Color"] == "Cinnamon"])
black_squirrel = len(data[data["Primary Fur Color"] == "Black"])

data_dictionary = {
    "Fur Color" : ["Gray", "Cinnamon", "Black"],
    "Count" : [grey_squirrel, cinnamon_squirrel, black_squirrel]
}

### CREATING A CSV FILE OF THE DATA SCRAPED ###
df = pandas.DataFrame(data_dictionary)
df.to_csv("squirrel_data.csv")