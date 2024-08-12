import streamlit as st
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

####################################################################
#Introducción

st.markdown("<h1>Drogas</h1>\n<h2>Cual es la mejor droga para tu caso?</h1>", unsafe_allow_html=True)
st.markdown("<p>&nbsp;&nbsp;&nbsp;Tenemos 3 tipos de drogas (X, Y, C) para controlar el colesterol. Dependiendo de las condiciones del paciente es que droga le hace mejor.</p>\n<p style='text-align: center; font-size: 20px;'><strong>Cual es la mejor droga para usted?</strong></p>", unsafe_allow_html=True)

df = pd.read_csv('drug200.csv')
st.dataframe(df.head())
st.dataframe(df.describe())
st.write(df.shape)

st.markdown("&nbsp;&nbsp;&nbsp; Para predecir la mejor droga vamos a utilizar un modelo ML llamado 'Decision Tree' el cual su precisión va aumentando con el incremento de pacientes en la base de datos", unsafe_allow_html=True)

####################################################################
#Preparando los datos

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

