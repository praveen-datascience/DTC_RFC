#Import All necessary libraries
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Read file
data = pd.read_csv("breast-cancer-wisconsin.data", skiprows=1, header=None)
data.head()
#set the column names 
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size' , 'Uniformity of Cell Shape' , 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin' , 'Normal Nucleoli' , 'Mitoses' , 'Class']
data.columns = column_names
data.head()
#sample code number is not helpful for prediction so will exclude 
new_data = data[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
 
'Mitoses', 'Class']]
new_data.head()

#Explore the data
new_data.describe()
#Data Pre processing
#check for null values in our data
new_data.isnull()

#Seaborn simplifies the process of detecting missing values
sb.heatmap(new_data.isnull())
#overview of our dataframe
new_data.info()

#object data type for the Bare Nuclei column will likely affect classification, and it's essential to address this before modeling
new_data['Bare Nuclei'] = pd.to_numeric(new_data['Bare Nuclei'], errors='coerce')
new_data.info()

#Seperate Dataframe into X and Y
X = new_data.values
y = new_data['Class'].values
#Delete the Target column Class (which is located at index 9) from X
X = np.delete(X,9,axis=1)

#Split the Dataset for Training and Testing
#70% Data willbe used for Training and 30% Data will be used for Testing
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Convert X_train and X_test back to DataFrames
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Handle missing values
X_train = X_train.fillna(X_train.mean())  # Fill missing values with mean
X_test = X_test.fillna(X_test.mean())

#Apply Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
# Make predictions on test data
dt_predictions = dt_classifier.predict(X_test)
# Evaluate model accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)

# Apply Random Forest Classifier
# check with 100 trees
rf_classifier = RandomForestClassifier(n_estimators=100)  
rf_classifier.fit(X_train, y_train)

# Make predictions on test data
rf_predictions = rf_classifier.predict(X_test)

# Evaluate model accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

