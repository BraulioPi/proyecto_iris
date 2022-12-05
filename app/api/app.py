from flask import Flask, request
import json
import psycopg2
import psycopg2.extras
import os
import pandas as pd
import time

print("API ACTIVADA")
# "motor://user:password@host:port/database"
#ojo aquí estamos metiendo las variables de ambiente (el archivo.env por medio del modulo os)
database_uri = f'postgresql://{os.environ["PGUSR"]}:{os.environ["PGPASS"]}@{os.environ["PGHOST"]}:5432/{os.environ["PGDB"]}'

app = Flask(__name__)
contador = 0
while contador <5:
    try:
        conn = psycopg2.connect(database_uri)
        print("conexión establecida")
        break
    except Exception as e:
        print("fallo conexión, volviendo a intentar")  
        contador = contador + 1
        time.sleep(5)  
 

@app.route('/') #esta madre genera la direccion hacia donde se manda el mensaje 
def home():
    cur = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
    cur.execute("select * from iris")
    results = cur.fetchall()
    cur.close()
    return json.dumps([x._asdict() for x in results], default=str)


#a partir de aqui falta modificar unas cosas
@app.route('/iris', methods=["POST", "GET", "DELETE", "PATCH"])
def florecita():
    if request.method == 'GET':
        cur = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        flor_id = request.args.get("id")
        cur.execute(f"""select 
                            * 
                        from 
                            iris 
                        where 
                            id={flor_id}""")
        results = cur.fetchone()
        cur.close()
        return json.dumps(results._asdict(), default=str)

    if request.method == "POST":
        flor = json.loads(request.data)
        print(flor)
        cur = conn.cursor()
        cur.execute(
            "insert into iris (id,sepallengthcm, sepalwidthcm , petallengthcm , petalwidthcm , species) values (%s,%s, %s, %s, %s, %s)",
            (flor[0]["id"],flor[0]["sepallengthcm"], flor[0]["sepalwidthcm"], flor[0]["petallengthcm"],flor[0]["petalwidthcm"], flor[0]["species"]),
        )
        conn.commit()
        #cur.execute("SELECT LASTVAL()")#muestra el ultimo valor de id
        flor_id = flor[0]["id"]
        cur.close()
        return json.dumps({"flor_id": flor_id})
    if request.method == "DELETE":
        cur = conn.cursor()
        flor_id = request.args.get("id")
        cur.execute(f"delete from iris where id={flor_id}")
        conn.commit()
        cur.close()
        return json.dumps({"flor_id": flor_id})
    if request.method == "PATCH":
        flor = json.loads(request.data)
        cur = conn.cursor()
        flor_id = request.args.get("id")
        cur.execute(
            "update iris set (sepallengthcm, sepalwidthcm , petallengthcm , petalwidthcm , species) = (%s,%s,%s,%s,%s) where id=%s ",
            (flor[0]["sepallengthcm"], flor[0]["sepalwidthcm"], flor[0]["petallengthcm"],flor[0]["petalwidthcm"], flor[0]["species"], flor_id),
        )
        conn.commit()
        cur.close()
        return json.dumps({"flor_id": flor_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)