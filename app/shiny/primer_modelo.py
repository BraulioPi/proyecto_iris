import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xgboost as xg
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import requests
import json
import pickle
####modulos
import wget
#estos son los necesarios para shiny
from shiny import App, render, ui, reactive



#variable de entorno para pruebas y por si acaso le pone el valor predeterminado
api_host  = os.getenv("API_HOST", "http://0.0.0.0:8080")#esot no conecta
#carga de tabla para establecer ciertos valores para input de entrada
def carga_data_completa(api_host):
    respuesta = requests.get(api_host+"/")
    data_raw_parametros  = respuesta.json()
    df_parametros        = pd.DataFrame.from_dict(pd.json_normalize(data_raw_parametros), orient="columns")
    return df_parametros
#imagenes necesarios para que funcione la app
def descarga_imagen(url_,nombre):
    wget.download(url_,out=nombre)
    print(f"imagen {nombre} descargada con exito")

df_parametros = carga_data_completa(api_host)    
print("APLICACION FUNCIONANDO")





                       
# ui de la app
app_ui = ui.page_fluid(
    # Adjust the styles to center everything
    ui.tags.style("#container {display: flex; flex-direction: column; align-items: center;}"),
    # Main container div
    ui.tags.div(
        ui.h2("Conjunto de datos Iris Setosa"),
        ui.row(
            ui.column(
            6,
            ui.h4("Historia"),
            ui.p("""
                El conjunto de datos flor Iris o conjunto de datos Iris Setosa de Fisher es un 
                conjunto de datos multivariante introducido por Ronald Fisher en su artículo de 1936,
                 The use of multiple measurements in taxonomic problems 
                 (El uso de medidas múltiples en problemas taxonómicos) 
                 como un ejemplo de análisis discriminante lineal. 
                 A veces, se llama Iris conjunto de datos de Anderson 
                 porque Edgar Anderson coleccionó los datos para cuantificar la variación 
                 morfológica de la flor Iris de tres especies relacionadas.
                 Dos de las tres especies se coleccionaron en la Península de la Gaspesia 
                 «todos son de la misma pastura, y recolectado el mismo día y medidos al mismo tiempo por la misma 
                 persona con el mismo aparato»."""
                 ),
                 ui.output_plot("imagen_florecita"),
            ui.p("""
                El conjunto de datos contiene 50 muestras de cada una de tres 
                especies de Iris (Iris setosa, Iris virginica e Iris versicolor). 
                Se midió cuatro rasgos de cada muestra: el largo y ancho del sépalo y pétalo, en centímetros. 
                Basado en la combinación de estos cuatro rasgos, Fisher desarrolló un modelo discriminante lineal 
                para distinguir entre una especie y otra."""
                ),
                ),
            ui.column(
            6,
            ui.h4("Qué puedes hacer con esta aplicación?"),
            ui.p("""
                HOLAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

                AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                """
                 )    
                )
        ),
        ui.h2("Datos y modificación"),
        ui.input_slider("inicio_tabla", "Selecciona a partir de que observacion quieres ver", min=0, max=150, value=0),
        ui.input_slider("fin_tabla", "Selecciona hasta que observacion quieres ver", min=0, max=150, value=10),
        ui.output_table("tabla_completa"),
        ui.input_numeric("id_tabla_filtro", "Escribe el id de la flor que quieres ver", value=10),
        ui.input_action_button("btn_tabla_filtro","busca la flor!"),
        ui.output_table("tabla_filtro_id"),
        ui.h2("Quieres enriquecer nuestra base de datos?, puedes meter una observacion"),
        ui.input_numeric("post_sepallengthcm", "Largo de sepalo", value=round(df_parametros["sepallengthcm"].mean(),1)),
        ui.input_numeric("post_sepalwidthcm", "Ancho de sepalo", value=round(df_parametros["sepalwidthcm"].mean(),1)),
        ui.input_numeric("post_petallengthcm", "Largo de petalo", value=round(df_parametros["petallengthcm"].mean(),1)),
        ui.input_numeric("post_petalwidthcm", "Ancho de petalo", value=round(df_parametros["petalwidthcm"].mean(),1)),
        ui.input_select(id="post_species", label="Especie de flor:", choices={"Iris-setosa":'Setosa' 
                                                                                ,"Iris-versicolor":'Versicolor'
                                                                                , "Iris-virginica":'Virginica'}),
        ui.input_action_button("btn_post","Enriquece la base!"),
        ui.output_text_verbatim("post_observacion"),
        ui.h2("Seleccion de modelo a entrenar"),
        ui.input_select(id="opcion", label="Modelo:", choices={"rf":"Random Forest","xgb":"XGBOOST"}),
        ui.input_action_button("btn","genera el modelo!"),
        ui.h2("Métricas y matriz de confusión del modelo "),
        ui.output_plot("viz"),
        ui.output_table("table_data"),
        ui.h2("Te gusta lo que ves?, entonces guarda el modelo"),
        ui.input_text("nombre_modelo", "Aqui pon el nombre del modelo", placeholder="AQUI PONES EL NOMBRE"),
        ui.input_action_button("btn_save","guardar modelo"),
        ui.output_text_verbatim("guarda_modelo"),
    id="container")
)






# logica del servidor, esta madre es el backend de la app
def server(input, output, session):

    @reactive.Calc
    @reactive.event(input.btn)
    def carga_procesa_data_train():
        diccionario_modelos = {"rf":RandomForestClassifier(random_state=0),
                                "xgb":xg.XGBClassifier(random_state =0)
                                }
        #cargando los datos desde la API
        respuesta = requests.get(api_host+"/")
        data_raw  = respuesta.json()
        df        = pd.DataFrame.from_dict(pd.json_normalize(data_raw), orient="columns")
        df.columns = [i.lower() for i in df.columns]

        #preprocesamiento de datos
        dic        = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        df["species_codi"]  = df["species"].map(dic)
        X                   = df.drop(["species","species_codi"],axis=1)#df[["sepallengthcm","petallengthcm"]]
        y                   = df["species_codi"]
        #division en train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
        #se hace fit de modelos
        modelo              = diccionario_modelos[input.opcion()]
        modelo.fit(X_train, y_train)
        #se genera validacion de prueba
        target_names        = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        #reporte de resultados
        y_pred              = modelo.predict(X_test)
        resultados          = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))
        #sacamos la matriz de confusion
        df_cm               = pd.DataFrame(confusion_matrix(y_test, y_pred),index=target_names,columns=target_names)

        return resultados, df_cm, modelo

    @output
    @render.text
    @reactive.event(input.btn_save)
    def guarda_modelo():
        _,_,modelo = carga_procesa_data_train()
        nombre = f"{input.nombre_modelo()}.pkl"
        pickle.dump(modelo,open(nombre,'wb'))
        return f"tu modelo: '{input.nombre_modelo()}' se ha guardado con éxito"
        
    # Chart logic
    @output
    @render.plot
    @reactive.event(input.btn)
    def viz():
        _, df_cm, _ = carga_procesa_data_train()
        fig, ax = plt.subplots()
        ax2     = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
        ax2.set(title="Matriz de confusion de modelo")
        return fig

    @output
    @render.plot
    def imagen_florecita():
        url_ = "https://camo.githubusercontent.com/bb83e831a860664959470e38c56bdce981c84687eafe04346d112be09a8c0227/68747470733a2f2f692e696d6775722e636f6d2f505171594761572e706e67"
        try:
            os.remove("florecitas.png")
            descarga_imagen(url_,nombre="florecitas.png")
            print("descargando imagen necesaria")
        except Exception as e:
            print("no hay imagen a borrar")  
            descarga_imagen(url_,nombre="florecitas.png")
            print("descargando imagen necesaria")  
        img  = mpimg.imread('florecitas.png')
        fig, ax = plt.subplots()
        ax2     = plt.imshow(img)
        plt.axis('off')
        
        return fig
    
    @output
    @render.table
    def tabla_completa():
        respuesta = requests.get(api_host+"/")
        data_raw  = respuesta.json()
        df        = pd.DataFrame.from_dict(pd.json_normalize(data_raw), orient="columns")
        df.columns = [i.lower() for i in df.columns]
        df         = df[input.inicio_tabla():input.fin_tabla()]

        return df

    @output
    @render.table
    @reactive.event(input.btn_tabla_filtro)
    def tabla_filtro_id():
        respuesta = requests.get(api_host+"/")
        data_raw  = respuesta.json()
        df        = pd.DataFrame.from_dict(pd.json_normalize(data_raw), orient="columns")
        if input.id_tabla_filtro() not in df["id"].unique():
            df =  pd.DataFrame(columns=["No existe flor con ese id :("])
        else:
            respuesta = requests.get(api_host+"/iris",{"id":input.id_tabla_filtro()})
            data_raw  = respuesta.json()
            df        = pd.DataFrame.from_dict(pd.json_normalize(data_raw), orient="columns")
        return df

    @output
    @render.text
    @reactive.event(input.btn_post)
    def post_observacion():
        #para hacer modificaciones
        df_post  = carga_data_completa(api_host)
        data_dic = {
            "id": int(df_post["id"].max()+1), 
            "sepallengthcm"  : input.post_sepallengthcm(),
            "sepalwidthcm"   : input.post_sepalwidthcm(),
            "petallengthcm"  : input.post_petallengthcm(), 
            "petalwidthcm"   :  input.post_petalwidthcm(),
            "species"        : input.post_species()
            }
        print(data_dic)
        id_guardado = data_dic["id"]
        data_dic    = json.dumps(data_dic)
        data_dic    = f"[{data_dic}]"
        requests.post(api_host+"/iris",data=data_dic)
           
        return f"tu flor se guardó con el id: '{id_guardado}'"

    @output
    @render.table
    def table_data():
        resultados, _ ,_= carga_procesa_data_train()
        return  resultados


# Connect everything
app = App(app_ui, server)

