from sklearn import datasets
import pandas as pd
import numpy as np
# generacion de datos sin tener que tenerlos en el repo
print("inicializando carga de datos IRIS")
iris = datasets.load_iris()
df   = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df["id"] = [i for i in range(0,len(df))]

df       = df[["id","sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)","target"]]
df["species"] = df["target"].map({0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'})
df            = df.drop("target",axis=1)
df.columns = ["id","sepallengthcm","sepalwidthcm","petallengthcm","petalwidthcm","species"]
df.to_csv("./data/iris.csv",index=False)
print("data inicial generada!!!!!!!!!")