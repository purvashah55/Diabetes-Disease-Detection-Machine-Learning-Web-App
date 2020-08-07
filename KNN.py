import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from flask import *
from sklearn.externals import joblib
import json
import os

from pandas.io.json import json_normalize
import numpy as np

dataset = pd.read_csv('dataset/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

# Split dataset into train and test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25,
                     random_state=42)

# import KNeighborsClassifier model
from sklearn.neighbors import KNeighborsClassifier as KNN




# train model


app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=["post"])
def predict():
    formvalues = request.form
    path1 = "/static/json/"
    with open(os.path.join(os.getcwd()+"/"+path1,'file1.json'), 'w') as f:
        json.dump(formvalues, f)
    with open(os.path.join(os.getcwd()+"/"+path1,'file1.json'), 'r') as f:
        values = json.load(f)
    df = pd.DataFrame(json_normalize(values))
    knn = KNN(n_neighbors=18)
    knn = KNeighborsClassifier(n_neighbors=18)
    knn.fit(X_train, y_train)

    saved_model = pickle.dumps(knn)
    knn_from_pickle = pickle.loads(saved_model);
    knn_from_pickle.predict(X_test);
    joblib.dump(knn, 'dm.pkl')

    model = joblib.load('dm.pkl')
    result = model.predict(df)
    a=np.array(1)
    if result.astype('int')==a.astype('int'):
        msg="Success"
    else:
        msg = "Unsuccess"
    positive_percent= model.predict_proba(df)[0][1]*100
    return render_template("index.html",msg=msg)



if __name__ == '__main__':
    app.run()

