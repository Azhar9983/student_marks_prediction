import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Student_Marks.csv")

#checking whether data is loaded or not 
# print(data.head(10))

# checking whether data is full or does it contain empty value
# print(data.isnull().sum())

# print(data["number_courses"].value_counts())

# figure1 = px.scatter(data_frame=data, x = "number_courses", y = "Marks", size = "time_study", title="Number of Courses and Marks Scored")
# figure1.show()

# figure2 = px.scatter(data_frame=data, x = "time_study", y = "Marks", size = "number_courses", title="Time Spent and Marks Scored", trendline="ols")
# figure2.show()

# correlation = data.corr()
# print(correlation["Marks"].sort_values(ascending=False))

x = np.array(data[["time_study","number_courses"]])
y = np.array(data["Marks"])

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain,ytrain)
# print(model.score(xtest,ytest))

feature = np.array([[8.25,8]])
print(model.predict(feature))


