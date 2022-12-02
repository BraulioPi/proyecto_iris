import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xg
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#estos son los necesarios para shiny
from shiny import App, render, ui, reactive

#primero la carga de datos 
#aqui va la conexion al sql (postgress)
df = pd.read_csv("/app/iris.csv").drop("id",axis = 1)
df.columns = [i.lower() for i in df.columns]


#preprocesamiento de datos
dic        = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

df["species_codi"]  = df["species"].map(dic)
X                   = df[["sepallengthcm","petallengthcm"]]
y                   = df["species_codi"]

#division en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

#por ahora se selecciona modelo(opciones de inputs)
opciones                         = {"rf":"Random Forest",
                                    "xgb":"XGBoost"}



# App UI - One input to select a ticker and two outputs for chart and table
app_ui = ui.page_fluid(
    # Adjust the styles to center everything
    ui.tags.style("#container {display: flex; flex-direction: column; align-items: center;}"),
    # Main container div
    ui.tags.div(
        ui.h2("Seleccion de modelo"),
        ui.input_select(id="opcion", label="Modelo:", choices=opciones),
        ui.output_plot("viz_scatter"),
        ui.output_plot("viz"),
        ui.output_table("table_data"),
    id="container")
)


# Server logic
def server(input, output, session):
    # Store data as a result of reactive calculation
    @reactive.Calc
    def modelo():
        diccionario_modelos = {"rf":RandomForestClassifier(random_state=0),
                               "xgb":xg.XGBClassifier()
        }
        target_names        = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        
        modelo              = diccionario_modelos[input.opcion()]
        modelo.fit(X_train, y_train)
        y_pred              = modelo.predict(X_test)
        
        resultados          = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))
        
        #sacamos la matriz de confusion
        df_cm               = pd.DataFrame(confusion_matrix(y_test, y_pred))
        
        return resultados, df_cm

    # Chart logic
    @output
    @render.plot
    def viz():
        _, df_cm = modelo()
        fig, ax = plt.subplots()
        ax2     = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
        ax2.set(title="Matriz de confusion de modelo")
        
        
        return fig
    
     # scatter plot de iris setosa
    @output
    @render.plot
    def viz_scatter():
        fig, ax = plt.subplots()
        ax2     = sns.scatterplot(x="sepallengthcm",y="petallengthcm",hue="species",data=df)
        ax2.set(title="Scatter")
        
        
        return fig


    # Table logic
    @output
    @render.table
    def table_data():
        resultados, _ = modelo()
        return  resultados


# Connect everything
app = App(app_ui, server)



