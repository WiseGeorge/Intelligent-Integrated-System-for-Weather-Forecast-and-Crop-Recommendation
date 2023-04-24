
import pandas as pd 
import numpy as npy
import tensorflow as tf
from keras import Sequential as sec
from keras import models 
import matplotlib.pyplot as plt

import time as t


class DataAnalytics:

    
    
    df = pd.read_excel("Datasets/TempFilter.xlsx")
    df_precip = pd.read_excel("Datasets/RainfallMean.xlsx")
  

    list_keys = list(df_precip.keys())
    list_precip = []
    for i in range (1,len(list_keys)-1):
        lista_precip = list_precip.extend(list(df_precip[list_keys[i]])) 

    
    list_Años = []
    list_Años = list(df_precip["Año"])*12

    @classmethod
    def makePrecipitationDf (cls):
        list_month = []
        for i in range (1,len(cls.list_keys)-1):
            for j in range (24):
                list_month.append(cls.list_keys[i])
        
        return pd.DataFrame({ "Mes": list_month,"Mean":cls.list_precip, "Año": cls.list_Años})
    
 

    @classmethod
    def Precipitacion_Anual(cls):
        
        return cls.df_precip[["Anual","Año"]]

    @classmethod
    def AnnualAvgAñoMax(cls,Año: int):
        """Devuelve el promedio de las temperaturas máximas registradas en el año 
        especificado por parámetro

        Args:
            Año (int)

        Returns:
            _type_: float
        """
        var = 0
        cont = 0
        for i in range (len(cls.df)):
            if cls.df["Año"][i] == Año:
                var += cls.df["Maxima Media"][i]
                cont +=1
        return (var/cont) 
    
    @classmethod
    def AnnualAvgAñoMin(cls,Año: int):
        """Devuelve el promedio de las temperaturas minimas registradas en el año 
        especificado por parámetro

        Args:
            Año (int)

        Returns:
            _type_: float
        """
        var = 0
        cont = 0
        for i in range (len(cls.df)):
            if cls.df["Año"][i] == Año:
                var += cls.df["Minima Media"][i]
                cont +=1
        return (var/cont) 
    
    @classmethod
    def AnnualAvgProvsegunAñoMax(cls, prov: str, Año: int):
        """Devuelve el promedio anual de la temperatura máxima de la provincia especificada por parámetro
        en el año especificado por parámetro

        Args:
            prov (str): 
            Año (int):

        Returns:
            _type_: float
        """
        var = 0
        cont = 0
        for i in range (len(cls.df)):
            if cls.df['Provincia'][i] == prov and cls.df["Año"][i] == Año:
                var += cls.df["Maxima Media"][i]
                cont +=1
        if cont !=0:
            return (var/cont)
        else:
            return 0
    
    @classmethod
    def AnnualAvgProvsegunAñoMin(cls, prov: str, Año: int):
        """Devuelve el promedio anual de la temperatura mínima de la provincia especificada por parámetro
        en el año especificado por parámetro

        Args:
            prov (str): 
            Año (int):

        Returns:
            _type_: float
        """
        var = 0
        cont = 0
        for i in range (len(cls.df)):
            if cls.df['Provincia'][i] == prov and cls.df["Año"][i] == Año:
                var += cls.df["Minima Media"][i]
                cont +=1
        return (var/cont)
    
    @classmethod
    def FullAnnualAvgAñoallProvMax(cls):
        """Devuelve el promedio anual de las temperaturas maximas de cada año registrado en la base de 
        datos

        Returns:
            _type_: DataFrame
        """
        lista = []
        listaaux = []
        for i in cls.df["Año"]:
            if i not in listaaux:
                lista.append(cls.AnnualAvgAñoMax(i))
                listaaux.append(i)
        dfresult = pd.DataFrame({"Años": listaaux, "Maxima Media" : lista })
        return dfresult
    
    @classmethod
    def FullAnnualAvgAñoallProvMin(cls):
        """Devuelve el promedio anual de las temperaturas mínimas de cada año registrado en la base de 
        datos

        Returns:
            _type_: DataFrame
        """
        lista = []
        listaaux = []
        for i in cls.df["Año"]:
            if i not in listaaux:
                lista.append(cls.AnnualAvgAñoMin(i))
                listaaux.append(i)
        dfresult = pd.DataFrame({"Años": listaaux, "Minima Media" : lista })
        return dfresult
    
    @classmethod
    def FullAnnualAvgProvsegunAñoMax(cls, Año: int ):
        """ 
        Devuelve la máxima media de temperaturas de 
        todas las provincias segun el año
        pasado por parámetro  
        
        Args: año(int)
        
        Return: list(int)
        """
        lista = []
        listaaux = []
        for i in range (len(cls.df)):
            if cls.df["Año"][i] == Año and cls.df["Provincia"][i] not in listaaux :
                lista.append(cls.AnnualAvgProvsegunAñoMax(cls.df["Provincia"][i], Año))
                listaaux.append(cls.df["Provincia"][i])
        dfresult = pd.DataFrame({"Provincias": listaaux, "Maxima Media": lista})
        return dfresult
    
    @classmethod
    def FullAnnualAvgProvsegunAñoMin(cls, Año: int ):
        """ 
        Devuelve la mínima media de temperaturas de 
        todas las provincias segun el año
        pasado por parámetro  
        
        Args: año(int)
        
        Return: list(int)
        """
        lista = []
        listaaux = []
        for i in range (len(cls.df)):
            if cls.df["Año"][i] == Año and cls.df["Provincia"][i] not in listaaux :
                lista.append(cls.AnnualAvgProvsegunAñoMin(cls.df["Provincia"][i], Año))
                listaaux.append(cls.df["Provincia"][i])
        dfresult = pd.DataFrame({"Provincias": listaaux, "Minima Media": lista})
        return dfresult
    
    @classmethod
    def FullAnnualAvgAñosegunProvMax(cls, prov: str):
        """ 
        Devuelve la máxima media de temperaturas de 
        de la povincia pasada por parámetro durante todos 
        los años que se alamacenan en BD
        
        Args: 
            año(int)
        
        Returns: 
            __type__: list
        """
        lista = []
        listaaux = []
        for i in range (len(cls.df)):
            if cls.df["Provincia"][i] == prov and cls.df["Año"][i] not in listaaux :
                lista.append(cls.AnnualAvgProvsegunAñoMax(prov, cls.df["Año"][i]))
                listaaux.append(cls.df["Año"][i])
        dfresult = pd.DataFrame({"Años": listaaux, "Maxima Media": lista})
        return dfresult
    
    @classmethod
    def FullAnnualAvgAñosegunProvMin(cls, prov: str):
        """ 
        Devuelve la mínima media de temperaturas de 
        de la povincia pasada por parámetro durante todos 
        los años que se alamacenan en BD
        
        Args: 
            año(int)
        
        Returns: 
            __type__: list
        """
        lista = []
        listaaux = []
        for i in range (len(cls.df)):
            if cls.df["Provincia"][i] == prov and cls.df["Año"][i] not in listaaux :
                lista.append(cls.AnnualAvgProvsegunAñoMin(prov, cls.df["Año"][i]))
                listaaux.append(cls.df["Año"][i])
        dfresult = pd.DataFrame({"Años": listaaux, "Minima Media": lista})
        return dfresult
    
    @classmethod
    def TotalAverageMax_Media(cls):
    
        list_Años = []
        list_prov = []
        list_max_med = []
        newdf = pd.DataFrame()
        cont= 0;
        for i in range (len(cls.df)) :
            if cls.df["Año"][i] not in list_Años:
                newdf.insert(cont,cls.df["Año"][i],cls.FullAnnualAvgProvsegunAñoMax(cls.df["Año"][i])["Maxima Media"],True)
                list_Años.append(cls.df["Año"][i])
                cont +=1
        for i in cls.df["Provincia"]:
            if i not in list_prov:
                list_prov.append(i)
        newdf.insert(0,"Provincias",list_prov,False)
        return newdf
   
    @classmethod
    def TotalAverageMin_Media(cls):
    
        list_Años = []
        list_prov = []
        list_max_med = []
        newdf = pd.DataFrame()
        cont= 0;
        for i in range (len(cls.df)) :
            if cls.df["Año"][i] not in list_Años:
                newdf.insert(cont,cls.df["Año"][i],cls.FullAnnualAvgProvsegunAñoMin(cls.df["Año"][i])["Minima Media"],False)
                list_Años.append(cls.df["Año"][i])
                cont +=1
        for i in cls.df["Provincia"]:
            if i not in list_prov:
                list_prov.append(i)
        newdf.insert(0,"Provincias",list_prov,False)
        return newdf

        
    @classmethod
    def AnnualAvgRainFallProvsegunAñoMax(cls, prov: str, Año: int):
        """Devuelve el promedio anual de la temperatura máxima de la provincia especificada por parámetro
        en el año especificado por parámetro

        Args:
            prov (str): 
            Año (int):

        Returns:
            _type_: float
        """
        var = 0
        cont = 0
        for i in range (len(cls.df)):
            if cls.df['Provincia'][i] == prov and cls.df["Año"][i] == Año:
                var += cls.df["Maxima Media"][i]
                cont +=1
        if cont !=0:
            return (var/cont)
        else:
            return 0
    

    @classmethod
    def PredictionAnnual(cls, x: str,y: str, listaux: list):
       


        ####################
        list =[]
        x_train = cls.df[x]
        y_train = cls.df[y]
        # model = sec()
        # model.add(tf.keras.layers.Dense(units = 67, input_shape = [1],input_dim=1))
        # model.save('C:\\Programacion\\Programas de python\\Machine Learning\\Proyecto Cientifco\\proyecto')
        
        model = sec(models.load_model('C:\\Programacion\\Programas de python\\Machine Learning\\Proyecto Cientifco\\proyecto'))
        model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['binary_accuracy'])
        
        model.fit(x_train, y_train, epochs = 24)
        Prediction = model.predict(listaux)
        t.sleep(0.1)
        for i in Prediction:
           list.append(i[0])
        dataframePredict = pd.DataFrame({'Años':listaux, y: list})
        return dataframePredict

