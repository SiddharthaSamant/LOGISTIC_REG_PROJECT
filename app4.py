import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model=pickle.load(open('heart1.pkl','rb'))








#stacked_averaged_models=pickle.load(open('stacked_averaged_models.pkl','rb'))


@app.route('/')
def home():
    return render_template('heart2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] 
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #prediction2 = model_lgb.predict(final_features)
    #prediction = stacked_averaged_models.predict(final_features)
    #prediction=mean()
    output = round(prediction[0], 3)
    

    return render_template('heart2.html', prediction_text='the person has {}00 percent chances of heart disease  '.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])


    output = prediction    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
