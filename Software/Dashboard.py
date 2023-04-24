import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import altair as alt
import time
from PIL import Image
from datetime import date
from Manager import DataAnalytics as da
from Graphics import DataVizualisation as DT


import requests, json
from datetime import datetime as d

# Models_Manager
from Models_Manager import CropWeather_Models
CW = CropWeather_Models()

rain_models_df = pd.read_excel('Models/Metrics/Rainfall_Models_Metrics.xlsx')

Temp_Model_dict = {
                    'Model': 'Linear Regression',
                    'MAE': 0.703,
                    'MSE': 0.941
}
temp_model_df = pd.DataFrame(Temp_Model_dict,index=[0])

crop_models_df = pd.read_excel('Models/Metrics/Crop_Models_Metrics.xlsx')


#Streamlit Config Functions
image = Image.open("Images/Icon.jpg")
st.set_page_config(page_title='Dashboard', page_icon = image, layout='wide')

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

users = {'Root': 'qwerty',
        'Wise_George': 'Jorgito16*',
        'Frankej': 'Frank07*'}

# Data Set
Temp_df = da.df
province_series = Temp_df["Provincia"]
Anual_Max_df = da.FullAnnualAvgAñoallProvMax()
Anual_Min_df = da.FullAnnualAvgAñoallProvMin()
RainFall_df = da.makePrecipitationDf()


#CleaningData
province_list = []
for element in province_series:
    if element not in province_list:
        province_list.append(element)

######################
# Page Title
######################




image = Image.open("Images/Earth2.jpg")

st.write("# **Intelligent Integrated System for Weather Forecast and Crop Recommendation.**")

st.image(image, use_column_width=True)
st.write('---')




with st.sidebar:
        
    selected = option_menu(
        menu_title = "Menu",
        options = ["Home", "Statistics", "Predict","Weather & Crop Software", "Download", "About"],
        icons = ["house","bar-chart-line-fill", "activity","window-dock","book", "envelope"],
        menu_icon = "image-alt ",
        default_index = 0,
        orientation = "vertical",
    )

    #st.header("⛅Real Time Weather Tracker")
    #st.code("Disable")
    # lista = weather("Cuba" +" weather")
    # st.write(lista[0])
    # #st.write(lista[1]+"h")
    # st.write(lista[2]) 
    # celsius =  (float(lista[3]) - 32) / 1.8
    # st.write(celsius, "°C")
    

if selected == "Home":
    
    # with st.sidebar:
    #     st.image("Sidebar.jpg")
    st.write("""
            ## Global Warming. Causes and Consequences. Study and Analysis of Climatological Variables in Cuba.

            """)

    col1, col2 = st.columns([2,2])

    with col1:
        st.header("Main Objective")
        st.write(""" 
                    ## ***Global Warming is a problem that affects the Planet Earth day by day.*** 
                    Having control over how,
                    as time passes, it increases; it is vitally important. 
                    The main objective is to show the reality of the incidence 
                    of this phenomenon in Cuba, as a result of the analysis of 
                    information and predictions, with the use of a methodology 
                    based on the learning of information and logic to study 
                    the data through the development of a descriptive and 
                    predictive analytical process. 
                    It is possible to deploy a set of software where it will 
                    be possible to visualize the result of the exploration 
                    of data based fundamentally on descriptive statistics that 
                    reflect the behavior of temperatures, demonstrating the potential
                    of ***data science*** and ***artificial intelligence*** in the extraction 
                    and processing of significant information.""")
    
        st.header("Direct impact of the effects of global warming on Cuba")
        st.write("The Planet Earth is a system in a very complex state of equilibrium, when creating a destabilization either to natural or artificial causes the earth triggers unpredictable natural disasters to equalize and maintain its balance, it can be considered an action-reaction process. These natural disasters can be severe both for entire civilizations, and for the fauna and flora of the planet. An impressive fact of great interest is that according to the Europa Press Data Agency, Cuba is today the number 1 country in displaced due to natural disasters.")
        st.image("Images/desplazado.jpg")
        
        st.header("Impact of global warming on the agricultural industry")
        st.write("Researchers and scientists have been warning for a long time that the increase in average temperature, the change in the amount and distribution of rainfall, the increase in the concentration of atmospheric carbon dioxide, are among the main effects of climate change on food production. As climate change increases, the areas for crops will decrease, unexpected changes will occur in the planting and harvest periods, an increase in pests and diseases will be evident and even some animal and vegetable species will become extinct. If the right decisions are not taken, a global food insecurity crisis can be triggered, being this even more acute in the countries with the least development of the agricultural industry.")
        st.write("According to the UN Food and Agriculture Organization (FAO), climate change will decrease productivity, stability and agricultural incomes in several areas that have already experienced high levels of food insecurity. That is why it is vitally important to increase world agricultural production by more than 70 percent in the next four decades, the only way to meet the food needs of the entire population.")
    
    with col2:

        st.header("Climate Change Infography by: ")
        st.write('***https://www.boldbusiness.com/***')
        image = Image.open("Images/Inphogra.png")
        st.image(image, use_column_width=True)

            
        st.header("The situation of soils in Cuba")
        st.write("At the Workshop Food Sovereignty and food plants for a better adaptation to climate change, held at the Friends of the Country Economic Society, Dr. Sergio Rodríguez Morales, director of the Tropical Viands Research Institute (INIVIT), precise:“In the country we have 76 percent of all agricultural areas with low productive soils, 14.9 percent are affected by salinity or modicity and 31 percent have low organic matter content” With the approach of Dr. Sergio Rodríguez Morales, it is reaffirmed that Cuba is not in a favorable position as far as fertile soils are concerned and if we add to this some of the environmental problems referred to above, the situation becomes critical.")
        st.image("Images/suelo.jpg")
    

if selected == "Statistics":
    ## 1. Display DataFrame
    st.header("📊DataSets Main Statistics")
    describe = DT.Display_Statics()
    if st.checkbox('Show Main Statistics', value = True):
        colum1, colum2 = st.columns(2)
        with colum1:
            st.subheader('Main Statistics Temperature')
            st.write(describe)
        with colum2:
            st.subheader("Main Statistics RainFall")
            st.write(DT.Rainfall_Describe())
        #st.download_button('Download Statics', data = a)

    col1,col2,col3 = st.columns([3,6,3])
    with col2:
        st.header("")
        st.header("Temperature Data Visualization")
        st.image("Images/temp.jpg")

    # 1. Anual Temperature Maxime and Minime
    colum1, colum2 = st.columns(2)
    with colum1:
        st.subheader("Maxime Average Per Year")
        st.bar_chart(data=Anual_Max_df, x="Años", y="Maxima Media")
        

    with colum2:
        st.subheader("Minime Average Per Year")
        st.bar_chart(data=Anual_Min_df, x="Años", y="Minima Media")
        

    ## 2. Display Graphics

    st.header("")
    years_to_filter = st.slider('years', 2006, 2020, 2007)  # min: 2006, max: 2020, default: 2007
    max_df = da.FullAnnualAvgProvsegunAñoMax(years_to_filter)
    min_df = da.FullAnnualAvgProvsegunAñoMin(years_to_filter)

    st.subheader("Max Average Per State")
    st.line_chart(data=max_df, x = "Provincias", y = "Maxima Media")
    st.subheader("Min Average Per State")
    st.line_chart(data=min_df, x = "Provincias", y = "Minima Media")

    st.header("")
    prov_to_filter = st.select_slider('Choose Province', province_list)
    Anual_Per_Prov_Max= da.FullAnnualAvgAñosegunProvMax(prov_to_filter)
    Anual_Per_Prov_Min= da.FullAnnualAvgAñosegunProvMin(prov_to_filter)
    st.bar_chart(data=Anual_Per_Prov_Max,x = "Años", y="Maxima Media" )
    
    st.header("")
    col1, col2, col3 = st.columns([3,12,3])
    with col2:
        st.subheader("Cross Relationship")
        st.pyplot(DT.Pair_Plot())

    st.header("")
    col1,col2,col3 = st.columns([3,5,3])

    with col2:
        st.header("Rainfall Data Visualization")
    st.image("Images/Rain.jpg")
    
    st.subheader("🌧️Rainfall Mean Average 🌧️")
    st.bar_chart(data = RainFall_df, y = "Mean", x = "Año")

if selected == "Predict":
    
    st.header("")
    st.header("")
    col1,col2,col3 = st.columns([1,18,1])
    with col2:
        st.header("Predictions Using Advanced Machine Learning Techniques")
    st.image("Images/ia2.jpg")
    st.header("")

    Year_list_before = [x for x in range(1900, 2000, 10)]
    Year_list_after = [x for x in range(2000, 2100, 10)]
    before = [28.519229888916016, 28.667667388916016, 28.816104888916016, 28.96453857421875, 29.112979888916016, 29.26140594482422, 29.409847259521484, 29.558284759521484, 29.706727981567383, 29.85515594482422]
    after = [30.00359344482422, 30.152034759521484, 30.30046844482422, 30.448909759521484, 30.59734344482422, 30.745779037475586, 30.894224166870117, 31.042648315429688, 31.191089630126953, 31.339527130126953]
    df_before = pd.DataFrame({"Años": Year_list_before, "Temperatura Media": before})
    df_after = pd.DataFrame({"Años": Year_list_after, "Temperatura Media": after})
    
    st.header("Temperature Predictions 2030-2070")
    temp =  [31.194957733154297, 31.246030807495117, 31.297096252441406, 31.348163604736328, 31.39923095703125]
    year = [2030, 2040, 2050, 2060, 2070]
    df_pred = pd.DataFrame({"Años": year, "Temperatura Media": temp})

    st.bar_chart(data = df_pred, x = "Años", y = "Temperatura Media" )

    st.header("")
    st.header("Increasing Temperatures during the 20th and 21st century")
    
    colum1, colum2, colum3 = st.columns([3,8,3])
    with colum2:
        st.write("## Average Temperatures 1900-2000")
    st.bar_chart(x = "Años", y = "Temperatura Media", data = df_before)
    
    colum1, colum2, colum3 = st.columns([3,8,3])
    with colum2:
        st.write("## Average Temperatures 2000-2100")
    st.bar_chart(x = "Años", y = "Temperatura Media", data = df_after)

    col1,col2,col3 = st.columns([2,10,2])
    with col2:
        st.subheader("***🌡️There is a variation of 2.8 Degrees Celsius from 1900 to 2090***")

if selected == "Weather & Crop Software":
    sel = option_menu(
        menu_title = "",
        options = ["Weather", "Crop", "Models Info"],
        icons = ["cloud-sun-fill", "flower1", "envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
    )


    
    if sel == 'Weather': 
        with st.sidebar:
            selec_w = st.selectbox('Weather Forecast', ['Temperature', 'Rainfall'])
        st.header('Cuban Weather Forecast Using Machine Learning Algoritms')
        st.write("""- The Model Predict the ***Average Temperature/Rainfall*** for the next **2 Centrys** in Each **Cuban States** 
        """)
        st.write("The Yearly Average Temperature in Cuba is 25.5 °C.")
        st.write("The Yearly Average Rainfall in Cuba is 112.9 mm.")

        col1, col2, col3 = st.columns([3,12,3])
        with col2:
            
            if selec_w == 'Temperature':
                st.write('### Temperature Model is Based on Linear Regression')
                st.write("The dataframe represent all possible models. The Model with the best ***Mean Absolute Error*** Achieved During the Model Evaluation is ***Support Vector Regression*** setted as default.")

                st.dataframe(temp_model_df[['Model','MAE','MSE']].style.highlight_min('MAE', color='indianred'),use_container_width=True)
        
                
                with st.expander('Select State/Year Values'):
                    # State Input
                    state = st.selectbox('State',CW.States)
                    index = CW.States.index(state)
            
                    # Year Input
                    this_year = date.today().year
                    year = st.selectbox('Year', range(this_year-20, this_year+200, +1))
                        
                    # Predict Buttom
                    pred_buttom = st.button('Temperature Prediction')
            
                if pred_buttom:
                    temp_pred = CW.Temp_Prediction(CW.Normalize_Weather_Models_Input(index, year))
                    temp_pred = round(temp_pred[0],2)
                    
                    st.write(temp_pred)
                    delta = '25.5 °C'
                    delta_inv = 'normal'
                    s = f'{temp_pred} °C'
                    if temp_pred > 25.5:
                        delta_inv = 'inverse'
                    
                    st.metric('🌡️Temperature Average', s, delta, delta_inv)

                    if temp_pred<=25.5:
                        st.write('#### ✅No temperature increase is predicted with respect to the annual average temperature in Cuba.')
                    else:
                        st.write(f"#### 📈⚠️In {year} in {state} an increase of {round(temp_pred-25.5,2)} degrees is predicted with respect to the average annual temperature in Cuba.")

            if selec_w == 'Rainfall':
                st.write("""### Rainfall Model is Based on Support Vector Regressor    
                """)
                st.write("""- ##### ***Support Vector Regression*** is the ***default model***""")
                st.write("The dataframe represent all possible models. The Model with the best ***Mean Absolute Error*** Achieved During the Model Evaluation is ***Support Vector Regression*** setted as default.")

                st.dataframe(rain_models_df[['Model','MAE']].style.highlight_min('MAE', color='indianred'),use_container_width=True)
                
                with st.expander('Select State/Year Values'):
                    
                    # Model Index => Support Vector Regressor: 2
                    model_index = 2

                    # State Input
                    state = st.selectbox('State',CW.States)
                    index = CW.States.index(state)
            
                    # Year Input
                    this_year = date.today().year
                    year = st.selectbox('Year', range(this_year-20, this_year+200, +1))

                    

                    # Other Models
                    m = st.checkbox('Use Others Models')
                    if m:
                        # Rainfall Models
                        model = st.radio('Models', CW.Rainfall_Models, horizontal=True)                        
                        model_index = CW.Rainfall_Models.index(model)

                    # Predict Buttom
                    pred_buttom = st.button('Rainfall Prediction')

                if pred_buttom:
                    rain_pred = CW.Rain_Prediction(CW.Normalize_Weather_Models_Input(index, year),model_index)
                    rain_pred = list(rain_pred.reshape(1,1))
                    rain_pred = round(rain_pred[0][0],2 )

                    s = f'{rain_pred} mm'
                    
                    delta_inv = 'normal'
                    delta = '50 mm'
                    if rain_pred < 50:
                        delta_inv = 'inverse'


                    st.metric('🌧️Rainfall Average', s, delta, delta_inv)
                    st.write('50 mm was defined as the limit between Low and High Rainfall Average')
                    
    if sel == 'Crop':
        selec_w = st.selectbox('Type', ['Manual Crop Recommendation', 'Semi-Automatic Crop Recommendation'])
        st.header('Crop Recommendation System Using Machine Learning Algoritms')
        st.write("""#### The system predicts recommended crops given climate and soil characteristics.  
        """)
        st.write('- ##### ***Manual:*** All climate and soil characteristics as inputs')
        st.write("""- ##### ***Semi-Automatic:*** Soil characteristics as input, but climate characteristics temperature and precipitation are fed from a predictive model. For humidity there are only two variants ***mean humidity*** and ***median humidity***""" )
        st.write('In The databases studied no absolute humidity values were recorded but relative humidity.')

        with st.sidebar:
            selec_c = st.selectbox('Crop Recommendation System', ['Manual', 'Semi-Auto'])

        if selec_c == 'Manual':
            st.write("""### Rainfall Model is Based on Support Vector Regressor    
            """)
            st.write("""- ##### ***Support Vector Regression*** is the ***default model***""")
            st.write("The dataframe represent all possible models. The Model with the best ***Mean Absolute Error*** Achieved During the Model Evaluation is ***Support Vector Regression*** setted as default.")

            st.dataframe(crop_models_df[['Models','Accuracy','Recall', 'F1_Score']].style.highlight_max(['Accuracy','Recall', 'F1_Score'], color='indianred'),use_container_width=True)
            
            with st.expander('Weather & Soil Characteristic'):
                
                # Model Index => Support Vector Regressor: 2
                model_index = 0

                # State Input
                st.subheader('Weather Values')
                temp = st.number_input('Temperature Value')
                rain = st.number_input('Rainfall Value')
                humidity = st.number_input('Humidity Value')
                st.write('---')

                    # State Input
                st.subheader('Soil Values ')
                ph = st.number_input('PH Value')
                n = st.number_input('N Value')
                p = st.number_input('P Value')
                k = st.number_input('K Value')
                st.write('---')

                # Predict Buttom
                pred_buttom = st.button('Crop Recommendation')

                if pred_buttom:
                    crop_pred = CW.Crop_Prediction_Each_Clf(CW.Normalize_Crop_Models_Input(temp,rain,humidity,ph,n,p,k))
                    dict_crop_pred = {
                            f'{CW.crop_models_names[0]}': crop_pred[0],
                            f'{CW.crop_models_names[1]}': crop_pred[1],
                            f'{CW.crop_models_names[2]}': crop_pred[2],
                            f'{CW.crop_models_names[3]}': crop_pred[3],
                            f'{CW.crop_models_names[4]}': crop_pred[4]
                    }
                
                    st.write(dict_crop_pred)
                    
        if selec_c == 'Semi-Auto':

            st.write("""### Rainfall Model is Based on Support Vector Regressor    
            """)
            st.write("""- ##### ***Support Vector Regression*** is the ***default model***""")
            st.write("The dataframe represent all possible models. The Model with the best ***Mean Absolute Error*** Achieved During the Model Evaluation is ***Support Vector Regression*** setted as default.")

            st.dataframe(crop_models_df[['Models','Accuracy','Recall', 'F1_Score']].style.highlight_max(['Accuracy','Recall', 'F1_Score'], color='indianred'),use_container_width=True)
            
            with st.expander('Weather & Soil Characteristic'):
                
                # Model Index => Support Vector Regressor: 2
                model_index = 0

                # State Input
                st.subheader('State/Year Values')
                # State Input
                state = st.selectbox('State',CW.States)
                index = CW.States.index(state)

                # Year Input
                this_year = date.today().year
                year = st.selectbox('Year', range(this_year-20, this_year+200, +1))

                
                temp = CW.Temp_Prediction(CW.Normalize_Weather_Models_Input(index, year))
                rain = CW.Rain_Prediction(CW.Normalize_Weather_Models_Input(index, year))

                col1,col2 = st.columns(2)
                with col1:
                    st.metric('Temperature', round(temp[0],2))
                with col2:
                    st.metric('Rainfall', round(rain[0], 2))

                humidity = st.selectbox('Humidity Value',['Humidity Mean', 'Humidity Median', 'Manual'])
                if humidity == 'Humidity Mean':
                    humidity = CW.humidity_mean
                    
                if humidity == 'Humidity Median':
                    humidity = CW.humidity_median
                    
                if humidity == 'Manual':
                    humidity = st.number_input('Humidity Value')
                
                st.metric('Humidity', round(humidity, 2))


                st.write('---')

                # State Input
                st.subheader('Soil Values ')
                ph = st.number_input('PH Value')
                n = st.number_input('N Value')
                p = st.number_input('P Value')
                k = st.number_input('K Value')
                st.write('---')

                # MultiModel
                mm = st.checkbox('Multi Model Recommendation', value=True)
                
                # Predict Buttom
                pred_buttom = st.button('Future Crop Recommendation')

                if pred_buttom:
                    if mm:
                        crop_pred = CW.Future_Crop_Prediction(index,year,humidity,ph,n,p,k)
                        dict_crop_pred = {
                                f'{CW.crop_models_names[0]}': crop_pred[0],
                                f'{CW.crop_models_names[1]}': crop_pred[1],
                                f'{CW.crop_models_names[2]}': crop_pred[2],
                                f'{CW.crop_models_names[3]}': crop_pred[3],
                                f'{CW.crop_models_names[4]}': crop_pred[4]
                        }
                        st.write('#### 🧺Crop Recommendation')
                        st.write(dict_crop_pred)
                    
                    else:
                        crop_pred = CW.Future_Crop_Prediction(index,year,humidity,ph,n,p,k,MultiPrediction=False)
                        st.write('#### 🧺Crop Recommendation')
                        st.write(f'##### {crop_pred}')








































if selected == "Download":
    #####################
    # Input Text Box
    #####################
            
            
    def convert_df(df):

        # IMPORTANT: Cache the conversion to prevent computation on every rerun

        return df.to_csv().encode('utf-8')



    #converting the sample dataframe

    #Importing All Data Sets
    temperature = pd.read_excel("../Datasets/TempFilter.xlsx")
    temp = convert_df(temperature)

    rainfall = da.makePrecipitationDf()
    rainfall = convert_df(rainfall)

    incremento = pd.read_csv("../Datasets/incremento.csv")
    incremento = convert_df(incremento)

    desplasado = pd.read_csv("../Datasets/desplazado.csv")
    desplasado = convert_df(desplasado)

    mediaMundial = pd.read_csv("../Datasets/mediamundial.csv")
    mediaMundial = convert_df(mediaMundial)

    st.header("Download the Main Data Sets")
    st.header("Source: ")
    st.subheader("http://www.onei.gob.cu/")
    st.subheader("https://www.epdata.es/")
    st.download_button( 
        label="Download Temperature Data as CSV",
        data=temp,
        file_name='Temperaturas.csv',
        mime='text/csv',
    )

    st.download_button( 
        label="Download Rainfall Data as CSV",
        data=rainfall,
        file_name='Rainfall.csv',
        mime='text/csv',
    )

    st.download_button( 
        label="Download Temperature Increase Data as CSV",
        data=incremento,
        file_name='Increase.csv',
        mime='text/csv',
    ) 
    
    st.download_button( 
        label="Download Global Mean Temperature Data as CSV",
        data=mediaMundial,
        file_name='Global_Mean.csv',
        mime='text/csv',
    )

    st.download_button( 
        label="Download Displaced people Data as CSV",
        data=desplasado,
        file_name='Displaced_people.csv',
        mime='text/csv',
    )


    


if selected == "About":

    col1,col2,col3 = st.columns([3,6,3])
    with col2:
        st.header("👨‍💻Research and development team👨‍💻: ")
        
        st.header("")
        
        st.header("Jorge Felix Martinez Pazos")
        st.subheader("Gmail: jorgito16040@gmail.com")
        st.subheader("Institution: ***Universidad de Ciencias Informaticas***")
        st.subheader("Research Center: ***Center of computational mathematics***")
        st.header("")
        st.header("Frank Enrique James Hernandez")
        st.subheader("Gmail: fjames07@gmail.com")
        st.subheader("Institution: ***Universidad de Ciencias Informaticas***")
        st.subheader("Research Center: ***Department of Artificial Intelligence***")
