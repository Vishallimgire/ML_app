import os
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pickle
import numpy as np

app = Flask(__name__)

classes = {
    'Carlsberg':[0., 0., 1., 0., 0., 0., 0., 0.] , 
    'Tuborg': [0., 0., 0., 0., 0., 0., 1., 0.], 
    '1664': [1., 0., 0., 0., 0., 0., 0., 0.], 
    'Brooklyn': [0., 1., 0., 0., 0., 0., 0., 0.] , 
    'Kongens': [0., 0., 0., 0., 1., 0., 0., 0.], 
    'Wiibroe': [0., 0., 0., 0., 0., 0., 0., 1.] ,
    'Jacobsen':[0., 0., 0., 1., 0., 0., 0., 0.] , 
    'Kronenbourg': [0., 0., 0., 0., 0., 1., 0., 0.]
}

@app.route('/', methods=['GET', 'POST'])
def image_match():
    if request.method == 'POST':
        resp = request.get_json().get('data')
        # selected_date = datetime.strptime(resp.get('selectedDate'), '%Y-%m-%d').date()
        # selected_date = selected_date.toordinal()
        selected_brand = classes.get(resp.get('brand_name'))
        loaded_model = pickle.load(open('O2_CO2_Beer_data.pkl', 'rb'))
        data = np.append(selected_brand, [resp.get('volumnHLT')])
        # pred = np.array([[0., 0., 1., 0., 0., 0., 0., 0.,737073, 506.088]])
        pred = np.array([data])
        result = loaded_model.predict(pred)
        # breakpoint()
        # print(loaded_model.score(result))
        return jsonify({'data': result[0]})
    
    elif request.method == 'GET':
        return render_template('Beer_Consumption.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)