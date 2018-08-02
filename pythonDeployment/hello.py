from flask import Flask, request
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
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
    modelword2vec = lib.word2vec	

    model_sentiment = lib.findSentiment(modelAns)
    act_sentiment = lib.findSentiment(actAns)
    
    avg_bench_mark = lib.run_avg_benchmark(modelAns, actAns,model=modelword2vec)  #need to put bin file to s3
    vector_dist =  lib.word_vectors.wmdistance(modelAns, actAns)   #need to install pyemd
    sementi_similarity = lib.semanticSimilarity(modelAns, actAns)

    ans = lib.callMe()
    #call sentense similarity function from main model
    return actAns + '   ' + modelAns + '   ' + str(model_sentiment) + '  ' + str(act_sentiment) + '   ' + str(avg_bench_mark) + '  ' + str(vector_dist) + '   ' + str(sementi_similarity)

def implModel(content):
	regr=joblib.load('regmodel.pkl')
	df2=pd.DataFrame(content,columns=['subjectid','EasyQuestions','AvgPerEasyQue','MediumQue','AvgPerMedQue','HardQuestions','AvgPerHardQue'])
	y_pred=regr.predict(df2)
	return str(y_pred[0])


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
