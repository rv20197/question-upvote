
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    int_features = [np.array(int_features)]
    prediction = model.predict(int_features)
    prediction = abs(prediction)
    
    output = round(prediction[0], 2)

    return render_template('index.html', 
                           prediction_text='Approximately upvotes for question will be: {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
