from flask import Flask, request
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
app = Flask(__name__)
import lib

@app.route("/")
def hello():
    return "Hello World2!"

@app.route('/predict/<nm>',methods = ['POST', 'GET'])
def login(nm):
   if request.method == 'POST':
       #content = request.form["name"]
       content = request.get_json(force=True)
       #print('sdasdasdasdasdas')
       return implModel(content)
       #return content['name']
   else:
       user = request.args.get('nm')
       return user


@app.route('/sen_sim/<sen>',methods = ['POST', 'GET'])
def sen_sim(sen):
    data = request.get_json(force=True)
    modelAns = data['modelAns']
    actAns = data['actAns']

    model_sentiment = lib.findSentiment(modelAns)
    act_sentiment = lib.findSentiment(actAns)
    
    avg_bench_mark = 0#lib.run_avg_benchmark(modelAns, actAns,model=word2vec)  #need to put bin file to s3
    vector_dist =  lib.word_vectors.wmdistance(modelAns, actAns)   #need to install pyemd
    sementi_similarity = lib.semanticSimilarity(modelAns, actAns)

    ans = lib.callMe()
    #call sentense similarity function from main model
    return actAns + '   ' + modelAns + '   ' + str(model_sentiment) + '  ' + str(act_sentiment) + '   ' + str(avg_bench_mark) + '  ' + str(vector_dist) + '   ' + str(sementi_similarity)


def implModel(content):
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    #using stored model
    regr = joblib.load('regmodel.pkl') 

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" 
        % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    #res = np.array_str(np.argmax(r2_score(diabetes_y_test, diabetes_y_pred))) 
    #str1 = ''.join(str(e) for e in r2_score(diabetes_y_test, diabetes_y_pred)[0])
    return str(r2_score(diabetes_y_test, diabetes_y_pred))
    # numpy.set_printoptions(threshold='nan')
    # reg = linear_model.LinearRegression()
    # reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    # return reg.coef_


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')