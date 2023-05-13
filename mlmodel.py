import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# from tensorflow import keras

df = pd.read_excel('FakeDataset_MusicGenres.xlsx')
df = df.drop(['Name'], axis = 1)
df = df.drop(['Guitar', 'Sitar', 'Violin', 'Sarod', 'Flute', 'Saxophone', 'Harmonium', 'Piano'], axis = 1)
df = df.drop(['Qawwali','Bhangra','Ghazal','Haweli Sangeet','Hindustani Classical', 'Carnatic Classical'], axis = 1)
df.rename(columns = {'EDM\n' : 'EDM', 'Pop\n': 'Pop'}, inplace = True)

# prediction using Linear Regression
# Separate the predictor variables (music genre ratings) and target variables (personality scores)
X = df[['Rap', 'EDM', 'Hip Hop', 'Metal', 'Pop', 'Rock', 'Jazz']]
y = df[['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']]
coloumnames_list = y.columns.to_list()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)



# Evaluate the model's accuracy using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)



# # Make a prediction for a new participant's personality scores based on their music genre ratings
new_participant_ratings = [3, 4, 2, 1, 5, 3, 4] # Example music genre ratings for a new participant
new_participant_scores = regressor.predict([new_participant_ratings])
print("Predicted personality scores for the new participant:", new_participant_scores)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))


# #Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
predictions = model.predict([[3, 4, 2, 1, 5, 3, 4]])

mydict = {}
for i in range(0,len(coloumnames_list)):
    mydict[coloumnames_list[i]] = predictions[0][i]

sorted_dict = dict(sorted(mydict.items(), key=lambda item: item[1]))




