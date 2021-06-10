import joblib
from flask import Flask, render_template, request
from helpers.dummies import *


app = Flask(__name__)
model = joblib.load('model3.h5')
scaler = joblib.load('scaler3.h5')


# weather = ['mist', 'rainy', 'snowy']
# weather_dummies = [0 for i in range(3)]
# try:
#     idx = weather.index(request.args['weather'])
#     weather_dummies[idx] = 1
# except:
#     pass



@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET'])
def predict():
    ''''
    data = [
        float(request.args['temp']),
        float(request.args['humidity']),
        int(request.args['hour']),
        int(request.args['is_rush_hour']),
        int(request.args['month'])
    ]
     '''

    data = [
        float(request.args['Gender']),
        float(request.args['Married']),
        float(request.args['Education']),
        float(request.args['Self_Employed']),
        int(request.args['ApplicantIncome']),
        int(request.args['CoapplicantIncome']),
        int(request.args['LoanAmount']),
        int(request.args['Loan_Amount_Term']),
        int(request.args['Credit_History'])
    ]

    data  += Dependents_dummies[request.args['Dependents']]

    data  += Property_Area_dummies[request.args['Property_Area']]

    prediction = round(model.predict(scaler.transform([data]))[0])
    x=" "
    if prediction ==1:
        x="able"
    else:
        x=" not able"

    return render_template('index.html', prediction_text='The customer is {} to repay the loan '.format(x))



if __name__ == "__main__":
    app.run(debug=True)