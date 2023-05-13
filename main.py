#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('form.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    coloumnames_list = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    predictions = model.predict(final_features)
    print(predictions)
    mydict = {}
    for i in range(0, len(coloumnames_list)):
        mydict[coloumnames_list[i]] = predictions[0][i]
    sorted_dict = dict(sorted(mydict.items(), key=lambda item: item[1]))
    Personality,PredictionValue = sorted_dict.popitem()
    return render_template('form.html', prediction_text='Your Personality is {} with value :{}'.format(Personality,PredictionValue) )

if __name__ == "__main__":
    app.run(debug=True)