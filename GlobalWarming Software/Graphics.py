import pandas as pd
import numpy as npy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from scipy import stats
from functions import TempManger as tm
from Manager import DataAnalytics as da


class DataVizualisation(object):

    Temp_df = pd.read_excel("../Datasets/TempFilter.xlsx")
    Max_Min_df = Temp_df[["Maxima Media","Minima Media"]]
    Maxima_Media_Serie = Temp_df["Maxima Media"]
    Mininma_Media_Serie = Temp_df["Minima Media"]
    
    Rainfall_df = da.makePrecipitationDf()


    #Prueba Se realizara eventualmente con DataFrame
    lista_annos = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    lista_Temp= [29, 30, 31, 32, 32.7, 32.9, 33, 33.6, 33.9, 34, 34.6, 35, 35.5, 36, 36.5,]
    lista_Province = ["Pinar del Rio", "La Habana", "Matanzas", "Villa Clara","Cienfuegos", "Sancti Spiritus","Ciego de Avila", "Camaguey", "Las Tunas", "Holguin", "Granma", "Guantanamo", "Isla de la Juventud" ]



    my_df = pd.DataFrame({'annos': lista_annos,
                        'Media Temp': lista_Temp})

    @classmethod
    def Display_Statics(cls):
        return cls.Temp_df[["Maxima Media","Minima Media"]].describe()
        
    @classmethod
    def Max_Min_Correlational_Graphic(cls):
        cls.Max_Min_df.corr()
        print("Correlacion De la Maxima Media y la Minima Media")
        graphic= sns.heatmap(cls.Max_Min_df.corr(), cmap = "magma", annot = True)
        return graphic

    @classmethod
    def Media_Temp_Anual(cls):
        
        graphic = sns.relplot(y = "Media Temp", x = "annos", data=cls.my_df, kind='line')
        plt.title('Temperatura Promedio Anual 2006-2020')
        plt.xlabel('Año')
        plt.ylabel('Temperatura Promedio')
        return graphic

    @classmethod
    def Media_Maxima_Provincial(cls):

        graphic = sns.relplot(data=cls.Temp_df, x= 'Provincia', y = 'Maxima Media', hue="Año", kind='line', palette='dark')
        graphic.set_xticklabels(rotation = 90)
        plt.title('Temperatura Promedio Anual Por Provicnias 2006-2020')
        plt.xlabel('Provincia')
        plt.ylabel('Temperatura Media')
        return graphic

    @classmethod
    def Media_Minima_Provincial(cls):

        graphic = sns.relplot(data=cls.Temp_df, x= 'Provincia', y = 'Minima Media', hue="Año", kind='line', palette='dark')
        graphic.set_xticklabels(rotation = 90)
        plt.title('Temperatura Promedio Anual Por Provicnias 2006-2020')
        plt.xlabel('Provincia')
        plt.ylabel('Temperatura Media') 
        return graphic

    @classmethod
    def max_media_Estacion(cls):
        catplot = sns.catplot(data = cls.Temp_df,x = "Provincia", y = "Minima Media", hue = "Estacion", palette='cool')

        #Rotation
        catplot.set_xticklabels(rotation = 90) 
        return catplot  

    @classmethod
    def Multi_Grph_Maxima(cls):
        mult_graph = sns.FacetGrid(cls.Temp_df, col = "Año")
        mult_graph.map(plt.hist, "Maxima Media")
        return mult_graph
    
    @classmethod
    def Multi_Grph_Minima(cls):
        mult_graph = sns.FacetGrid(cls.Temp_df, col = "Año")
        mult_graph.map(plt.hist, "Minima Media")
        
    @classmethod
    def Pair_Plot(cls):
        sns.set_theme(context='notebook', style='darkgrid', palette='summer', font='sans-serif', font_scale=0.5, color_codes=True, rc=None)
        graphic = sns.pairplot(data = cls.Temp_df, hue='Año', palette='winter')
        
        return graphic

    @classmethod
    def Promedio_Anual_Provincial(cls,provincia: str):
        prov_df = tm.FullAnnualAvgAnnosegunProvMax(provincia)
        bar_Plot=sns.regplot(data = prov_df, x = "Años", y = "Maxima Media")
        return bar_Plot

    @classmethod
    def fff(cls):
        
        df = tm.Total_DataFrame()

        rel_plot=sns.relplot(data=df, x= 'Provincia', y = 'Maxima Media', hue="Año", kind='line', palette='dark')
        rel_plot.set_xticklabels(rotation = 90)
        plt.title('Temperatura Promedio Anual Por Provicnias 2006-2020')
        plt.xlabel('Provincia')
        plt.ylabel('Temperatura Media')
        return rel_plot

    @classmethod
    def Rainfall_Describe(cls):
        return cls.Rainfall_df["Mean"].describe()

    @classmethod
    def Rainfall_Dataframe(cls):
        return da.makePrecipitationDf()
        

