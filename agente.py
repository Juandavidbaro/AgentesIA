import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Falta la clave de OpenAI. Asegúrate de haberla configurado en el archivo .env")

#https://www.kaggle.com/datasets/carrie1/ecommerce-data/data
# Cargar la base de datos con langchain
db = SQLDatabase.from_uri("sqlite:///ecommerce.db")

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

# Crear la cadena con la LLM 
cadena = SQLDatabaseChain(llm=llm, database=db, verbose=True)

formato = """
Dada una pregunta del usuario:
1. Crea una consulta de SQLite3.
2. Revisa los resultados.
3. Devuelve el dato.
4. Si tienes que hacer alguna aclaración o devolver cualquier texto, siempre en español.
#{question}
"""

def consulta(input_usuario):
    consulta_sql = formato.format(question=input_usuario)

    print("Consulta SQL generada:", consulta_sql)
    
    resultado = cadena.invoke(consulta_sql) 
    return resultado

if __name__ == "__main__":
    while True:
        #Cuantos productos hay en la base de datos?
        #Cuales son los 3 paises en los que tenemos más ventas totales? Saca una tabla con el nombre del país y el total de ventas.
        #Dame los 5 productos más vendidos en cantidad, en una tabla que tenga el nombre del producto, el total de unidades vendidas y el porcentaje sobre el total de unidades de todos los productos.
        pregunta = input("Por favor, ingrese su consulta en lenguaje natural (escriba 'salir' para terminar): ")
        if pregunta.lower() == 'salir':
            break
        respuesta = consulta(pregunta)
        print("Respuesta:", respuesta['result'])
