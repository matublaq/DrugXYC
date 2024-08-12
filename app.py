import streamlit as st
import streamlit.components.v1 as components
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
####################################################################
####################################################################
#Variable para retraer o desplegar de los botones
if 'show_content' not in st.session_state:
    st.session_state['show_content'] = False

#Función para alternar el estado
def toggle_content():
    st.session_state['show_content'] = not st.session_state['show_content']
####################################################################
####################################################################


####################################################################
#Introducción

st.markdown("<h1>Drogas</h1>\n<h2>Cual es la mejor droga para tu caso?</h1>", unsafe_allow_html=True)
st.markdown("<p>&nbsp;&nbsp;&nbsp;Tenemos 5 tipos de drogas (X, Y, A, B, C) para controlar el colesterol. Dependiendo de las condiciones del paciente es que droga le hace mejor.</p>\n<p style='text-align: center; font-size: 20px;'><strong>Cual es la mejor droga para usted?</strong></p>", unsafe_allow_html=True)

df = pd.read_csv('drug200.csv')
st.dataframe(df.head())
st.dataframe(df.describe())
st.write(df.shape)

st.markdown("<p style='font-size: 25px;'>&nbsp;&nbsp;&nbsp; Para predecir la mejor droga vamos a utilizar un modelo de ML llamado <strong style='color: blue;'>Decision Tree</strong>.</p>", unsafe_allow_html=True)
if st.button("¿Cómo Funciona el Modelo de Árbol de Decisión?", on_click=toggle_content()):
    if st.session_state['show_content']:
        st.markdown('''
&nbsp;&nbsp;&nbsp;Un árbol de decisión es un modelo de machine learning utilizado para tareas de clasificación y regresión. Se llama así porque su estructura se asemeja a un árbol, donde cada nodo representa una decisión basada en una característica de los datos, y cada rama representa un resultado posible de esa decisión. Al final del árbol, las hojas representan las predicciones finales.

#### 1. **Estructura del Árbol**
- **Nodo Raíz**: Es el punto de partida del árbol, que contiene todos los datos.
- **Nodos de Decisión**: Cada uno representa un punto de decisión donde el conjunto de datos se divide en subconjuntos más pequeños basados en una característica específica.
- **Hojas**: Estos son los nodos terminales del árbol. Cada hoja representa una predicción (una clase en el caso de clasificación, o un valor numérico en el caso de regresión).

#### 2. **Cómo Crece un Árbol de Decisión**

1. **Selección de la Mejor Característica**:
   - El algoritmo comienza en el nodo raíz, donde selecciona la característica que mejor divide los datos según un criterio específico. Los criterios más comunes son:
     - **Entropía e Información Ganada**: La entropía mide la impureza en un conjunto de datos. El árbol selecciona la característica que reduce más la entropía, es decir, la que maximiza la información ganada.
     - **Índice Gini**: Es otra métrica de impureza. El árbol selecciona la característica que minimiza la impureza Gini en los subconjuntos resultantes.

2. **División de los Datos**:
   - Los datos se dividen en subconjuntos basados en la característica seleccionada y un valor umbral. Cada rama del nodo de decisión representa un subconjunto de datos.
   
3. **Repetición del Proceso**:
   - El proceso de selección y división se repite de forma recursiva en cada nodo de decisión, creando nuevos nodos y ramas hasta que se cumple una condición de parada.

#### 3. **Criterios de Parada**
El crecimiento del árbol se detiene cuando:
- **Profundidad Máxima**: Se alcanza un número máximo de niveles en el árbol.
- **Número Mínimo de Muestras en un Nodo**: Un nodo no tiene suficientes muestras para seguir dividiéndose.
- **Impureza Mínima**: La impureza en un nodo es lo suficientemente baja.

#### 4. **Predicción**
- Para predecir con un árbol de decisión, se toma una nueva muestra y se sigue el camino del árbol desde la raíz hasta una hoja, basándose en los valores de las características de la muestra.
- La hoja en la que termina la muestra contiene la predicción del modelo.

#### 5. **Ventajas y Desventajas**

- **Ventajas**:
  - **Fácil de interpretar**: Los árboles de decisión son intuitivos y transparentes.
  - **Manejo de datos mixtos**: Pueden manejar tanto características categóricas como numéricas.
  - **Relaciones no lineales**: Capturan relaciones complejas entre características.

- **Desventajas**:
  - **Sobreajuste**: Pueden ajustarse demasiado a los datos de entrenamiento si no se limitan adecuadamente.
  - **Inestabilidad**: Son sensibles a pequeñas variaciones en los datos.
  - **Menor precisión**: Otros modelos, como Random Forests o Gradient Boosting, suelen ofrecer una mejor precisión.

### Resumen
&nbsp;&nbsp;&nbsp;Los árboles de decisión son herramientas poderosas en machine learning. Es importante utilizarlos con cuidado para evitar problemas como el sobreajuste, y considerar técnicas de mejora como la poda o los métodos de ensemble para obtener mejores resultados.
''', unsafe_allow_html=True)

####################################################################
#Preparando los datos
page_url = 'tree.html'
if st.button("Ver código del modelado de los datos"):
    with open(page_url, 'r') as file:
        html_content = file.read()
    components.html(html_content, height=600, scrolling=True)


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder() #Convierte etiquetas de texto en valores numéricos
le_sex.fit(['F', 'M']) #LabelEncoder aprenderá que 'F' se mapea a 1 y 'M' se mapea a 0 (o viceversa)
X[:, 1] = le_sex.transform(X[:, 1]) #Transforma los valores usando LabelEncoder. Convierte 'F' y 'M' en sus correspondientes valores numéricos

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

y = df['Drug']

#Separando los datos en entrenamiento y testeo
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

####################################################################
#Modelo
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=5) #entropy=Medida de impureza o desorden en el conjunto da datos: 'Es el criterio que usará para la división del conjunto de datos en cada nodo.'. max_depth=4: Cantidad de niveles maximo del árbol.
drugTree.fit(X_train, y_train) #Ajusta el modelo a los datos de entrenamiento. Es decir, el árbol de decisión aprende patrones a partir de los datos de entrenamiento.

#Predicción
predTree = drugTree.predict(X_test)

#Evaluación
from sklearn import metrics
st.markdown(f"<p>La presicion del modelo es del {round((metrics.accuracy_score(y_test, predTree))*100, 2)}%</p>", unsafe_allow_html=True) #((predicciones correctas)/(total predicciones))*100%

#Grafico
fig, ax = plt.subplots()
tree.plot_tree(drugTree)
st.pyplot(fig)

####################################################################
#Predicción de la mejor droga para tu caso
st.markdown("<h2>¿Cuál es la mejor droga para tu caso?</h2>", unsafe_allow_html=True)
st.markdown("<p>Para predecir la mejor droga para tu caso, ingresa los siguientes datos:</p>", unsafe_allow_html=True)

# Datos personales
age = st.slider("Age", min_value=0, max_value=117, value=50)
sex = st.radio("Sex", ["M", "F"])
bp = st.radio("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
cholesterol = st.radio("Cholesterol", ["NORMAL", "HIGH"])
na_to_k = st.number_input("Na_to_K", min_value=0.0, value=16.5)

sex_encoded = le_sex.transform([sex])[0]
bp_encoded = le_BP.transform([bp])[0]
cholesterol_encoded = le_Chol.transform([cholesterol])[0]

# Predicción
if bp == "NORMAL" and cholesterol == "NORMAL":
    st.markdown("<p style='font-size: 25px;color: green;'>No necesitas tomar ninguna droga. ESTAS JOYA PÁ</p>", unsafe_allow_html=True)
else:
  input_data = np.array([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]])
  prediction = drugTree.predict(input_data)[0]
  st.markdown(f"<p style='font-size: 25px;'>La mejor droga para usted es: <strong>{prediction}</strong></p>", unsafe_allow_html=True)
    

