# FASE 1 IMPORTAMOS 

import pandas as pds 
import numpy as nmp 
import matplotlib.pyplot as plt
import seaborn as sbn 
 

df = pds.read_csv("./data/prestamos_knn_1000.csv")



# Fase 2: Exploración de datos y limpieza.
df.info()


#Quitamos los valores nulos
df = df.dropna(how='any')
print(df.isnull().sum())


df['ingreso_mensual_cop'] = pds.to_numeric(df['ingreso_mensual_cop'], errors='coerce')
df['monto_prestamo_cop'] = pds.to_numeric(df['monto_prestamo_cop'], errors='coerce')


""" print(df.head(20))
print((df['monto_prestamo_cop'] < 0) & (df['ingreso_mensual_cop'] < 0)) """


df_new = df.drop_duplicates()
df_new.info()
print(df_new.head(20))


df_new.dropna(inplace=True, how='any')
df_new.to_csv("./data/prestamos_knn_1000_limpio.csv", index=False)

df_new.describe()