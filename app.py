import pandas as pd
import numpy as np
from flask import Flask,request,app,jsonify,url_for,render_template
import pickle

# make a flask app
app = Flask(__name__)


# loading all the files

tfidf = pickle.load(open('tfidf_fitted.pkl','rb'))
svc = pickle.load(open('svc.pkl', 'rb'))
dec_tree = pickle.load(open('dec_tree.pkl', 'rb'))
gnb = pickle.load(open('gnb.pkl', 'rb'))
mnb = pickle.load(open('mnb.pkl', 'rb'))
log_reg = pickle.load(open('log_reg.pkl', 'rb'))

# for redirecting to the home page

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
   #it takes the only one string in the json

    #standardizing the input data , changing to the numerical vector
    new_data =tfidf.transform([data])

    #predicing the output ->feeding the transformed input
    output = log_reg.predict(new_data)

    #return 2d array [[30.4]], need for first value only
    return jsonify(output[0])

# @app.route('/predict', methods = ['POST'])
# def predict():
#     data = [float(x) for x in request.form.values()] #forgot the () gave an error
#     final_input  = scaler.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output = regmodel.predict(final_input)[0]
#     return render_template("home.html", prediction_text = "The Predicted House price is {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)
# //this app.py is running in the system python env so that's why scikit learn was not installed and was giving the error when run it
# // now installed the scikit-learn 

# //Method Not Allowed
# The method is not allowed for the requested URL. if the pasted the api in the url 
# is accessed through postman only