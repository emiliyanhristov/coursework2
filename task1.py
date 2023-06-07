import pandas as pd
from sklearn.ensemble import RandomForestClassifier #import the RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import csv

model = RandomForestClassifier(n_estimators=100) #creating a RandomForestClassifier with 100 trees

#loading the training data
train_data = pd.read_csv("TrainingDataBinary.csv")

#loading the testing data
test_data = pd.read_csv("TestingDataBinary.csv")

#splitting the training data into two sets
#x containing all of the features
#y containing 0 and 1, the labels of the events (0 - normal, 1 - threat)
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

#splitting the training data into 80% train and 20% test data to help the AI model learn
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

#fitting the RandomForestClassifier model
model.fit(x_train, y_train)

#using the trained model to make predictions on the test data
predictions = model.predict(test_data)

#converting the predictions to list
predictionsList = predictions.tolist()

#function to write the predicted values into the test data file
def addPredictedValues():
    #index to remember the element of the list that should be added to the row 
    insert_index = -1
    #reading the testing file using csv reader
    with open('TestingDataBinary.csv', 'r') as input_file:
        reader = csv.reader(input_file)
        data = list(reader)

    #iterating over rows in data
    for row in data:
        #inserting the prediction from the prediction list into the row
        if insert_index < 0:
            #adding column 129 to the column labels
            row.append(129)
        else:
            row.append(predictionsList[insert_index])
        
        #increasing the index     
        insert_index += 1

    #writing the modified data back into the testing file
    with open('TestingDataBinary.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)

#calling the addPredictedValues() function        
addPredictedValues()
