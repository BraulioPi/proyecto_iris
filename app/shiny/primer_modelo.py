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
import glob
import wget
import datetime
#estos son los necesarios para shiny
from shiny import App, render, ui, reactive


print("generando conexión de aplicación shiny")
#variable de entorno para pruebas y por si acaso le pone el valor predeterminado
api_host  = os.getenv("API_HOST", "http://0.0.0.0:8080")
print("calculando variables para funcionamiento correcto")

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

#cuantiles necesarios para hacer mapeo de funcionalidad beta    
def calcula_cuantiles_beta(df_parametros):
    lista_col = ['sepallengthcm', 'sepalwidthcm', 'petallengthcm', 'petalwidthcm']
    base_df = pd.DataFrame()
    base_df["index"] = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    for i in lista_col:
        pivot = df_parametros[i].quantile([.1,.2,.3,.4,.5,.6,.7,.8,.9,1]).reset_index()
        base_df = base_df.merge(pivot,on=["index"],how='left')
    base_df["respuesta"] = (base_df["index"] * 10).astype(int)
    base_df              = base_df.drop("index",axis=1)
    return base_df    

#pipeline de procesamiento para funcionamiento...
df_parametros   = carga_data_completa(api_host)    
df_cuantiles    = calcula_cuantiles_beta(df_parametros)
dict_cuantiles  = df_cuantiles.set_index(df_cuantiles["respuesta"]).drop("respuesta",axis=1)

print("APLICACION FUNCIONANDO")





                       
# ui de la app
app_ui = ui.page_fluid(
    # Adjust the styles to center everything
    ui.tags.style("#container {display: flex; flex-direction: column; align-items: center;}"),
    # Main container div
    ui.tags.div(
        ui.h1("Conjunto de datos Iris Setosa"),
        ui.row(
            ui.column(
            6,
            ui.h2("Historia"),
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
            ui.h2("¿Qué puedes hacer con esta aplicación?"),
            ui.p("""
                 Con esta aplicación puedes ver los datos, generar modelos de machine learning, 
                 guardar modelos y probar nuestraa nueva funcionalidad Beta.
                """
                 ),  
            ui.p("""
                 Esta aplicación tiene 3 partes principales: Datos y modificación, juguemos con machine learning!
                 y nuestra funcionalidad Beta:
                """
                 ),
            ui.tags.ul(
                ui.tags.li(
                    """
                    Datos y modificación: Aquí puedes ver nuestros datos de florecitas,
                    buscar alguna flor por su identificador, agregar una flor si es que tienes información,
                    corregir alguna flor que pienses que tiene mal sus datos y eliminar alguna flor si lo crees necesario.
                    """
                ),
                ui.tags.li(
                    """
                    Juguemos con Machine Learning!: Te permite entrenar un modelo de Machine Learning con los datos, 
                    guardar el modelo si te gustan los resultados, ver que modelos has guardado y predecir con el modelo
                    que selecciones con los datos que quieras.
                    """
                ),
                ui.tags.li(
                    """
                    Funcionalidad Beta: Nuestro equipo de desarrollo tiene que comer, así que esta funcionalidad permite
                    que puedas seguir disfrutando de este servicio. Lo unico que tienes que hacer, es darnos tus datos :)
                    """
                )
            ),
        ui.output_plot("imagen_musculosa"),
        ),
        ),
        ui.h1("Datos y modificación"),
        ui.output_plot("imagen_comarca"),
        ui.row(
            ui.column(
            6,

            ui.h2("Veamos los datos. Cada flor tiene un id que la identifica."),
            ui.input_slider("inicio_tabla", "Selecciona a partir de que observacion quieres ver", min=0, max=150, value=0),
            ui.input_slider("fin_tabla", "Selecciona hasta que observacion quieres ver", min=0, max=150, value=10),
            ui.output_table("tabla_completa"),
            ui.h2("Busca una flor por medio de su id!"),
            ui.input_numeric("id_tabla_filtro", "Escribe el id de la flor que quieres ver", value=10),
            ui.input_action_button("btn_tabla_filtro","busca la flor!"),
            ui.output_table("tabla_filtro_id"),

            ui.h2("¿Crees que alguna observación está mal?. Puedes removerla escribiendo el id de la flor a eliminar."),
            ui.input_numeric("id_delete", "Escribe el id de la flor que quieres borrar", value=0),
            ui.input_action_button("btn_delete","Elimina la flor!"),
            ui.output_text_verbatim("delete_observacion"),
            ),
            ui.column(
            6,

            ui.h2("¿Quieres enriquecer nuestra base de datos?. Puedes meter una observación."),
            ui.input_numeric("post_sepallengthcm", "Largo de sepalo", value=round(df_parametros["sepallengthcm"].mean(),1)),
            ui.input_numeric("post_sepalwidthcm", "Ancho de sepalo", value=round(df_parametros["sepalwidthcm"].mean(),1)),
            ui.input_numeric("post_petallengthcm", "Largo de petalo", value=round(df_parametros["petallengthcm"].mean(),1)),
            ui.input_numeric("post_petalwidthcm", "Ancho de petalo", value=round(df_parametros["petalwidthcm"].mean(),1)),
            ui.input_select(id="post_species", label="Especie de flor:", choices={"Iris-setosa":'Setosa' 
                                                                                    ,"Iris-versicolor":'Versicolor'
                                                                                    , "Iris-virginica":'Virginica'}),
            ui.input_action_button("btn_post","Enriquece la base!"),
            ui.output_text_verbatim("post_observacion"),
            
            ui.h2("También puedes corregir los datos de una flor por medio de su id"),
            ui.input_numeric("id_patch", "Escribe el id de la flor que quieres corregir", value=0),
            ui.input_numeric("patch_sepallengthcm", "Largo de sepalo", value=0),
            ui.input_numeric("patch_sepalwidthcm", "Ancho de sepalo", value=0),
            ui.input_numeric("patch_petallengthcm", "Largo de petalo", value=0),
            ui.input_numeric("patch_petalwidthcm", "Ancho de petalo", value=0),
            ui.input_select(id="patch_species", label="Especie de flor:", choices={"Iris-setosa":'Setosa' 
                                                                                    ,"Iris-versicolor":'Versicolor'
                                                                                    , "Iris-virginica":'Virginica'}),
            ui.input_action_button("btn_patch","corrige la flor!"),                                                                        
            ui.output_text_verbatim("patch_observacion"),
            )
        ),      

        ui.h1("Juguemos con Machine Learning"),
        ui.output_plot("imagen_juguemos"),
        ui.row(
            ui.column(
            6,
            ui.h2("Entrena un modelo!"),
            ui.input_select(id="opcion", label="Modelo:", choices={"rf":"Random Forest","xgb":"XGBOOST"}),
            ui.input_action_button("btn","genera el modelo!"),
            ui.h2("Métricas y matriz de confusión del modelo "),
            ui.output_plot("viz"),
            ui.output_table("table_data")
                      ) 
                    ,
        
            ui.column(
             6, 
            ui.h2("¿Te gusta lo que ves?. Entonces guarda el modelo."),
            ui.input_text("nombre_modelo", "Aqui pon el nombre del modelo", placeholder="AQUI PONES EL NOMBRE"),
            ui.input_action_button("btn_save","guardar modelo"),
            ui.output_text_verbatim("guarda_modelo"),

            ui.h2("¿Quieres ver que modelos has guardado?. Aprieta el botón de buscar modelos"),
            ui.input_action_button("btn_muestra_modelos","Buscar modelos!"),
            ui.output_table("tabla_modelos"),

            ui.h2("Carga un modelo para probar metiendo datos!"),
            ui.input_text("nombre_modelo_prueba", "Aqui pon el nombre del modelo que quieres cargar", placeholder="AQUI PONES EL NOMBRE"),
            ui.h2("Mete datos y veamos que dice el modelo!"),
            ui.input_numeric("predict_sepallengthcm", "Largo de sepalo", value=0),
            ui.input_numeric("predict_sepalwidthcm", "Ancho de sepalo", value=0),
            ui.input_numeric("predict_petallengthcm", "Largo de petalo", value=0),
            ui.input_numeric("predict_petalwidthcm", "Ancho de petalo", value=0),
            ui.input_action_button("btn_carga_predict","Predice con el modelo!"),
            ui.output_text_verbatim("carga_predice")   
            )
        ),

        ui.h1("BETA: ¿Qué flor eres de acuerdo a tu personalidad?"),
        ui.output_plot("imagen_beta"),
        ui.h3(" Selecciona del 1 al 10 que tan de acuerdo estás con la afirmación (1 es nada y 10 es muy de acuerdo)"),
        ui.h5("Ojo: No guardamos datos, solo se los vendemos a algunos patrocinadores que respetan tu privacidad...."),
        ui.h2("No te causa tristeza ver perros callejeros"),
        ui.input_slider("pregunta_1", "", min=1, max=10, value=1),
        ui.h2("La pizza debe llevar piña a fuerza, para más placer "),
        ui.input_slider("pregunta_2", "", min=1, max=10, value=1),
        ui.h2("Eres de los raritos que piensan que R es mejor que Python"),
        ui.input_slider("pregunta_3", "", min=1, max=10, value=1),
        ui.h2("Tienes problemas cognitivos, para ti las quesadillas no deben llevar queso"),
        ui.input_slider("pregunta_4", "", min=1, max=10, value=1),
        ui.input_action_button("btn_beta","Dime que flor soy!"),
        ui.output_plot("imagen_tipo_flor"),
        ui.output_text_verbatim("muestra_tipo_flor"),
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

        #preprocesamiento de datos
        dic        = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        df["species_codi"]  = df["species"].map(dic)
        X                   = df.drop(["species","species_codi","id"],axis=1)#df[["sepallengthcm","petallengthcm"]]
        y                   = df["species_codi"]
        #division en train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
        #se hace fit de modelos
        modelo              = diccionario_modelos[input.opcion()]
        modelo.fit(X_train, y_train)
        #se genera validacion de prueba
        target_names        = ['setosa', 'versicolor', 'virginica']
        #reporte de resultados
        y_pred              = modelo.predict(X_test)
        resultados          = pd.DataFrame(classification_report(y_test, y_pred, target_names=target_names, output_dict=True))
        #sacamos la matriz de confusion
        df_cm               = pd.DataFrame(confusion_matrix(y_test, y_pred),index=target_names,columns=target_names)

        return resultados, df_cm, modelo

    @reactive.Calc
    @reactive.event(input.btn_beta)
    def tipo_flor():
        df        = df_parametros.copy()
        #preprocesamiento de datos
        dic         = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        dic_reverse = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
        df["species_codi"]  = df["species"].map(dic)
        X                   = df.drop(["species","species_codi","id"],axis=1)#df[["sepallengthcm","petallengthcm"]]
        y                   = df["species_codi"]
        #se hace fit de modelo para funcionalidad beta
        modelo              = xg.XGBClassifier(random_state =0)
        modelo.fit(X, y)
        X_test              = pd.DataFrame()
        #se mapea la respuesta al rango de las varaibles
        X_test["sepallengthcm"] = [dict_cuantiles["sepallengthcm"][input.pregunta_1()]]
        X_test["sepalwidthcm"]  = [dict_cuantiles["sepalwidthcm"][input.pregunta_2()]]
        X_test["petallengthcm"] = [dict_cuantiles["petallengthcm"][input.pregunta_3()]]
        X_test["petalwidthcm"]  = [dict_cuantiles["petalwidthcm"][input.pregunta_4()]] 
        #reporte de resultados
        y_pred                  = modelo.predict(X_test)
        pred                    = dic_reverse[y_pred[0]]
        return pred    

    @output
    @render.text
    @reactive.event(input.btn_beta)
    def muestra_tipo_flor():
        pred   = tipo_flor()
        mensaje = {'Iris-setosa':"""
                                 Todo muy bien, te consideras un buen ciudadano y lo eres.
                                 Eres alguien muy flexible y triunfador. Saludos jaja.
                                 Tienes petalos cortos.
                                 
                                 """,
                   'Iris-versicolor':"""
                                     Eres una persona que está aprendiendo del mundo, sigue así.
                                     Normalmente te molesta que la gente haga sonidos cuando come.
                                     Tienes petalos medianos.
                                     """,
                   'Iris-virginica': """
                                     Claramente eres una persona que no le tiene respeto a dios,
                                     ¿Cómo que prefieres R a Python por deus?. Esperamos que algún
                                     día retomes el camino, te recomendamos ir a terapia. Tienes petalos largos.
                                     """
                   }
        mensaje = mensaje[pred]
        return f"""Según nuestro poderoso modelo de I.A, eres una {pred}! 
                  {mensaje} """   

    @output
    @render.plot
    @reactive.event(input.btn_beta)
    def imagen_tipo_flor(): 
        pred   = tipo_flor()
        flor   = pred.split("-")[1]
        img    = mpimg.imread(f'./images/{flor}.png')
        fig, ax = plt.subplots()
        ax2     = plt.imshow(img)
        plt.axis('off')
        
        return fig     
    
    @output
    @render.table
    @reactive.event(input.btn_muestra_modelos)
    def tabla_modelos():
        try:
            lista_modelos = glob.glob('./*.pkl')
            lista_tiempos = []
            df_modelos    = pd.DataFrame();
            df_modelos["direccion_modelo"] = lista_modelos
            df_modelos["nombre_modelo"]    = [i.replace(".pkl","").replace("./","") for i in df_modelos["direccion_modelo"]]
            for i in lista_modelos:lista_tiempos.append(datetime.datetime.fromtimestamp(os.path.getctime(i)))
            df_modelos["fecha creación"]   = lista_tiempos
            df_modelos                     = df_modelos.drop("direccion_modelo",axis=1)
            if len(df_modelos) ==0:
                df_modelos  = pd.DataFrame(columns=["no has generado ningun modelo :("])
        except Exception as e:
            df_modelos = pd.DataFrame(columns=["no has generado ningun modelo :("])
            
        return df_modelos

    @output
    @render.text
    @reactive.event(input.btn_save)
    def guarda_modelo():
        _,_,modelo = carga_procesa_data_train()
        nombre = f"{input.nombre_modelo()}.pkl"
        pickle.dump(modelo,open(nombre,'wb'))
        return f"tu modelo: '{input.nombre_modelo()}' se ha guardado con éxito"

    @output
    @render.text
    @reactive.event(input.btn_carga_predict)
    def carga_predice():
        nombre_modelo_origin = input.nombre_modelo_prueba()
        nombre_modelo = f"{nombre_modelo_origin}.pkl"
        dic           = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
        try:
            #cargamos el modelo
            modelo_cargado = pickle.load(open(nombre_modelo, "rb"))
            #generamos nuestro dataset para predecir
            df_pred        = pd.DataFrame()
            df_pred["sepallengthcm"] = [input.predict_sepallengthcm()]
            df_pred["sepalwidthcm"] =  [input.predict_sepalwidthcm()]
            df_pred["petallengthcm"] = [input.predict_petallengthcm()]
            df_pred["petalwidthcm"] =  [input.predict_petalwidthcm()] 
            #predecimos clase 
            prediccion     = modelo_cargado.predict(df_pred)
            #tenemos que predecir probabildiad tambien 
            proba          = str(round(modelo_cargado.predict_proba(df_pred)[0][prediccion[0]],2)*100)[0:4]
            pred           = dic[prediccion[0]]
            return f"Modelo {nombre_modelo_origin} predijo que es de la clase {pred} con una probabilidad del {proba} %!"
        except Exception as e:
            print(e)
            return "No hay ningun modelo guardado con ese nombre :("
            
        
        
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
        img  = mpimg.imread('./images/florecitas.png')
        fig, ax = plt.subplots()
        ax2     = plt.imshow(img)
        plt.axis('off')
        
        return fig
    @output
    @render.plot
    def imagen_beta(): 
        img  = mpimg.imread('./images/tipo.png')
        fig, ax = plt.subplots()
        ax2     = plt.imshow(img)
        plt.axis('off')
        
        return fig    

    @output
    @render.plot
    def imagen_musculosa(): 
        img  = mpimg.imread('./images/musculosa.png')
        fig, ax = plt.subplots()
        ax2     = plt.imshow(img)
        plt.axis('off')
        
        return fig

    @output
    @render.plot
    def imagen_juguemos(): 
        img  = mpimg.imread('./images/juguemos.png')
        fig, ax = plt.subplots()
        ax2     = plt.imshow(img)
        plt.axis('off')
        
        return fig   

    @output
    @render.plot
    def imagen_comarca(): 
        img  = mpimg.imread('./images/comarca.png')
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
    @render.text
    @reactive.event(input.btn_patch)
    def patch_observacion():
        id_filtro = int(input.id_patch())

        respuesta = requests.get(api_host+"/")
        data_raw  = respuesta.json()
        df        = pd.DataFrame.from_dict(pd.json_normalize(data_raw), orient="columns")

        if id_filtro not in df["id"].unique():
            mensaje = "no hay ninguna flor con ese id :("
        else:
            data_dic = {
                "sepallengthcm"  : input.patch_sepallengthcm(),
                "sepalwidthcm"   : input.patch_sepalwidthcm(),
                "petallengthcm"  : input.patch_petallengthcm(), 
                "petalwidthcm"   : input.patch_petalwidthcm(),
                "species"        : input.patch_species()
                }
            print(data_dic)
            data_dic = json.dumps(data_dic)
            data_dic = f"[{data_dic}]"
            requests.patch(api_host+"/iris",data = data_dic,params = {"id":id_filtro})
            mensaje  = f"La flor con el id:'{id_filtro}' se modificó con éxito"
           
        return mensaje    

    @output
    @render.text
    @reactive.event(input.btn_delete)
    def delete_observacion():
        id_filtro = int(input.id_delete())

        respuesta = requests.get(api_host+"/")
        data_raw  = respuesta.json()
        df        = pd.DataFrame.from_dict(pd.json_normalize(data_raw), orient="columns")
        if id_filtro not in df["id"].unique():
            mensaje = "no hay ninguna flor con ese id :("
        else:
            requests.delete(api_host+"/iris",params = {"id":id_filtro})
            mensaje = f"La flor con el id: '{id_filtro}' se ha removido con éxito"
        return mensaje    

    @output
    @render.table
    def table_data():
        resultados, _ ,_= carga_procesa_data_train()
        return  resultados


# Connect everything
app = App(app_ui, server)