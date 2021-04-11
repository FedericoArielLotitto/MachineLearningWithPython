##--------pruebas sobre los datos de clima--------##

import numpy as np
from  sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
## importado de datos desde "Clima.csv"

datos_de_clima = pd.read_csv('Clima.csv', usecols = ['MinTemp', 'RISK_MM'])
datos_de_clima = datos_de_clima.sort_values(by = 'MinTemp')
print(datos_de_clima)

##ENTENDIENDO LA DATA

print("Informacion en el dataset")

print(datos_de_clima.keys())

'''
    SEGÚN EL ANÁLISIS, LOS DATOS PRESENTADOS EN EL DOCUMENTO CSV CONTIENEN DATOS ACERCA DE UN AÑO DE 366 DIAS.
    ESTOS DATOS ABARCAN LAS COLUMNAS, QUE SON FACTORES METEOROLÓGICOS DE DESCRIPCION DE CLIMA.
    ESTOS SON:
    'MinTemp' (TEMPERATURA MINIMA DEL DIA) ,
    'MaxTemp' (TEMPERATURA MAXIMA DEL DIA),
    'Rainfall' (VOLUMEN DE LLUVIA),
    'Evaporation'(EVAPORACION),
    'Sunshine'(HORAS DE SOL POR DIA),
    'WindGustDir' (DIRECCION DE VIENTO FUERTE),
    'WindGustSpeed'(VELOCIDAD DE VIENTO FUERTE),
    'WindDir9am'(VELOCIDAD DEL VIENTO NORMAL A LAS 9 AM),
    'WindDir3pm'(DIRECCION DEL VIENTO NORMAL A LAS 3PM),
    'WindSpeed9am'(VELOCIDAD A LAS 9 AM DEL VIENTO NORMAL),
    'WindSpeed3pm' (VELOCIDAD DEL VIENTO NORMAL A LAS 3PM),
    'Humidity9am'(VAPOR DE AGUA EN ATMOSFERA A LAS 9 AM),
    'Humidity3pm'(VAPOR DE AGUA EN ATMOSFERA A LAS 3 PM),
    'Pressure9am'(PRESION ATMOSFERICA A LAS 9 AM),
    'Pressure3pm'(PRESION ATMOSFERICA A LAS 3 PM),
    'Cloud9am' (VOLUMEN DE AGUA ACUMULADO -NUBES- A LAS 9 AM),
    'Cloud3pm'(VOLUMEN DE AGUA ACUMULADO A LAS 3PM),
    'Temp9am'(TEMPERATURA ATMOSFERICA A LAS 9 AM),
    'Temp3pm'(TEMPERATURA ATMOSFERICA A LAS 3 PM),
    'RainToday'(SI HUBO LLUVIA EL DIA EN PARTICULAR -FILAS DEL DOCUMENTO-),
    'RISK_MM'(RIESGO DE LLUVIA EN MILIMETROS),
    'RainTomorrow'(SI LLOVIO O NO AL DIA SIGUIENTE)
'''
''' EN ESTE CASO, INTENTAREMOS APLICAR EL MECANISMO DE REGRESION SIMPLE A PARTIR DE LOS DATOS INGRESADOS DE TEMPERATURA MINIMA
    PARA PREDECIR EL RIESGO DE AGUA EN MM DEL DIA '''

print("Temperatura Minima (VARIABLE INDEPENDIENTE)\n", datos_de_clima.MinTemp, "\nRiesgo en mm (VARIABLE DEPENDIENTE)\n", datos_de_clima.RISK_MM)

#---VARIABLE INDEPENDIENTE---#
X = datos_de_clima.iloc[:, :1].values #.reshape(-1,1)

#---VARIABLE DEPENDIENTE---#
y = datos_de_clima.iloc[:, 1].values #.reshape(-1,1)

print(X)
print(y)

print(datos_de_clima.shape)
print(datos_de_clima.describe())

##---fuente para graficos----
font = {'family': 'serif','color':  'darkred','weight': 'normal','size': 16}
##-------------

#---------------------------------GRAFICO DE DATOS DEL CLIMA (RIESGO EN FUNCION DE LA TEMPERATURA MINIMA-----------------------------------#
datos_de_clima.plot(x='MinTemp', y='RISK_MM')
plt.scatter(X, y, color = 'blue')
plt.title("Riesgo en mm en funcion de la temperatura minima")
plt.xlabel("Temperatura Minima", fontdict = font)
plt.ylabel("Riesgo en mm", fontdict = font)
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------#

##-----------------------------------MISSING VALUES---------------------------------------------------------------------------------------##
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
imputer = imputer.fit(X)
X= imputer.transform(X)
#-------------------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------SEPARACION DE DATOS EN CONJUNTO DE ENTRENAMIENTO Y DE TEST---------------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#-------------------------------------------------------------------------------------------------------------------------------------------#


##-------ALGORITMO DE REGRESION LINEAL-------------------------------##

from sklearn.linear_model import LinearRegression
regression = LinearRegression()

regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicción': y_pred.flatten()})
#-------------------------------------------------------------------------------##

#-------------------------GRAFICO DE CONJUNTO DE ENTRENAMIENTO: VARIABLE INDEPENDIENTE EN FUNCION DE SU PREDICCCION----------------------------#
plt.figure("CONJUNTO DE ENTRENAMIENTO")
plt.scatter(X_train, y_train, color = '#FF6347')
plt.plot(X_train, regression.predict(X_train), color = '#20B2AA')
plt.title("Riesgo_MM en funcion de Temp.Min.(Conjunto de Entrenamiento)")
plt.xlabel("MinTemp", fontdict = font)
plt.ylabel("RISK_MM", fontdict = font)
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------#


#----------GRAFICO DE CONJUNTO DE TESTING: VARIABLE INDEPENDIENTE DE TESTING EN FUNCION DE LA PREDICCION DEL CONJUNTO DE ENTRENAMIENTO---------#
plt.figure("CONJUNTO DE TESTING")
plt.scatter(X_test, y_test, color = '#FF6347')
plt.plot(X_train, regression.predict(X_train), color = '#20B2AA')
plt.title("Riesgo_MM en funcion de Temp.Min.(Conjunto de Testing)")
plt.xlabel("MinTemp", fontdict = font)
plt.ylabel("RISK_MM", fontdict = font)
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------#


print("\nPRECISION DEL ALGORITMO DE REGRESION")
print(regression.score(X_train, y_train),"\n")


#-----ERRORES------#      
from sklearn import metrics
print('Error Medio Absoluto (MAE):', metrics.mean_absolute_error(y_test, y_pred))  
print('Error cuadrático medio (MSE):', metrics.mean_squared_error(y_test, y_pred, squared=True))  
print('Error cuadrático medio de raíz (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#------------------#


'''
    CONCLUSION A PARTIR DE REGRESION LINEAL SIMPLE:
        A PARTIR DEL GRAFICO DE ENTRENAMIENTO Y DE TEST, PUEDE OBSERVARSE QUE LOS DATOS 
	SON MUY DISPERSOS PARA EL ALGORITMO. EN ESTE SENTIDO, PUEDE VERIFICARSE EN SU 
	PRECISION CASI IGUAL A 0, QUE SU PRECISION EN LA PREDICCION DE DATOS ES MUY BAJA.
        EN ESTE SENTIDO, IMPLICA QUE DENTRO DEL MODELO, PREDECIR A PARTIR DE LA TEMPERATURA 
	MINIMA DE LOS DIAS DEL AÑO EL RIEGO EN MM DEL DIA MEDIANTE REGRESION LINEAL SIMPLE 
	PARA DATOS CON ESTE NIVEL DE DISPERSION, NO ES LO MAS CONVENIENTE PARA ESTE TIPO DE 
	DATOS.
        A PARTIR DEL ERROR CUADRATICO MEDIO, PUEDE OBSERVARSE LA DIFERENCIA EN REGRESION 
	LINEAL ENTRE LOS DATOS ESPERADOS Y LOS DATOS DEPREDICCION: LOS DATOS DISPERSOS 
	HACEN VER EL SESGO SURGIDO DE LOS DATOS.
        A PARTIR DEL ERROR CUADRATICO MEDIO DE RAIZ, PUEDE OBSERVARSE QUE NO ES MUY 
	CONVENIENTE PARA ESTOS DATOS EL MODELO DE REGRESION LINEAL SIMPLE (ESTA MUY 
	LEJANO AL CERO).
'''



