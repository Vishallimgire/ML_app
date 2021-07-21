import os
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pickle
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 2)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
loaded_model = pickle.load(open('combine_model_polynomial.pkl', 'rb'))

df = px.data.tips()
X = df.total_bill.values[:, None]
X_train, X_test, y_train, y_test = train_test_split(
    X, df.tip, random_state=42)

models = {'Polynomial Regression': linear_model.LinearRegression,
        #   'Decision Tree': tree.DecisionTreeRegressor,
        #   'k-NN': neighbors.KNeighborsRegressor
          }

def format_coefs(coefs):
    equation_list = [f"{coef}x^{i}" for i, coef in enumerate(coefs)]
    equation = "$" +  " + ".join(equation_list) + "$"

    replace_map = {"x^0": "", "x^1": "x", '+ -': '- '}
    for old, new in replace_map.items():
        equation = equation.replace(old, new)

    return equation


app.layout = html.Div([
    # html.P("Select Model:"),
    dcc.Dropdown(
        id='model-name',
        # options=[{'label': x, 'value': x} 
        #          for x in models],
        value='Regression',
        clearable=False,
        disabled=True,
        style={'display':'None'}
    
    ),
    dcc.Graph(id="graph"),
])

# 
@app.callback(
    dash.dependencies.Output("graph", "figure"), 
    [dash.dependencies.Input('model-name', "value")]
    )
def train_and_display(name):
    dataset = pd.read_csv('combine_data.csv')
    dataset = dataset[['TotalVolumeHLT', 'EnergyUsedAdjusted']]
    dataset = dataset.dropna()
    dataset = dataset[dataset['EnergyUsedAdjusted'] != 0]
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    poly_reg.fit(X)
    X_poly = poly_reg.transform(X)
    x_range_poly = poly_reg.transform(x_range)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    y_poly = model.predict(x_range_poly)
    # print(x_range.squeeze())
    equation = format_coefs(model.coef_.round(2))
    # x_range = np.linspace(X.min(), X.max(), 100)
    # y_range = loaded_model.predict(poly_reg.fit_transform(x_range))
    fig = go.Figure([
        go.Scatter( x=X.squeeze(), y=y,
                   name='train', mode='markers'),
        # go.Scatter(x=X_test.squeeze(), y=y_test, 
        #            name='test', mode='markers'),
        go.Scatter(x=x_range.squeeze(), y=y_poly, 
                   name='prediction')
    ])
    fig.update_layout(
    title="Polynomial Regression",
    xaxis_title="TotalVolumeHLT",
    yaxis_title="EnergyUsedAdjusted"
    )
    return fig


brand_type = {
        'Læsk':[0., 1., 0.],
         'Cider':[1., 0., 0.],
          'Øl':[0., 0., 1.]
}

energy_type = {
        'CO2':[1., 0., 0., 0.], 
        'Electricity':[0., 1., 0., 0.],
        'Water':[0., 0., 0., 1.],
        'Heat':[0., 0., 1., 0.]
}

     
@server.route('/home', methods=['GET', 'POST'])
def image_match():
    print('hi')
    if request.method == 'POST':
        resp = request.get_json().get('data')
        # selected_date = datetime.strptime(resp.get('selectedDate'), '%Y-%m-%d').date()
        # selected_date = selected_date.toordinal()
        selected_brand = brand_type.get(resp.get('brand_type'))
        selected_energy = energy_type.get(resp.get('energy_type'))
        data = np.append(selected_brand, selected_energy)
        data =   np.append(data, [resp.get('volumnHLT')])
        # pred = np.array([[0., 0., 1., 0., 0., 0., 0., 0.,737073, 506.088]])
        pred = np.array([data])
        result = loaded_model.predict(poly_reg.fit_transform(pred))
        # breakpoint()
        # print(loaded_model.score(result))
        return jsonify({'data': result[0]})
    
    elif request.method == 'GET':
        print('page loading')
        return render_template('Beer_Consumption.html')


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5000, debug=False)