import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from clean_data import data_final
app = Flask(__name__)
model = joblib.load(open('tweets_model.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tweet = request.form.values()
    tweet = ' '.join(tweet)

    print(tweet)
    #final_features = [np.array(int_features)]
    final_features = data_final(tweet)
    # Use the trained model to predict the probability of the tweet belonging to each class
    prediction = model.predict_proba(final_features)
    print(prediction[0, 1])
    threshold = 0.4  # You can adjust this threshold based on your model's performance
    predicted_label = 1 if prediction[0, 1] >= threshold else 0
    print(predicted_label)
    
    # Print the predicted label
    prediction_class ="Negative"
    if predicted_label == 1:
         prediction_class ="Positive"
    
    return render_template('index.html', prediction_text='Ce tweet est ' + str(prediction_class))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)