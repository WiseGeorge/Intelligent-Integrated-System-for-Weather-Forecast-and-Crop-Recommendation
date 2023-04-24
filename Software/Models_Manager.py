from joblib import load
import numpy as np
import pandas as pd

class CropWeather_Models():

    def __init__(self) -> None:

        # Class
        self.Crop_Label = ['apple','banana','blackgram','chickpea','coconut','coffee',
                      'cotton','grapes','jute','kidneybeans','lentil','maize','mango',
                      'mothbeans','mungbean','muskmelon','orange','papaya','pigeonpeas',
                      'pomegranate','rice','watermelon']
        
        self.States = ['Camagüey', 'Ciego de Ávila', 'Cienfuegos', 'Granma', 'Guantánamo',
                    'Holguín', 'Isla de la Juventud', 'La Habana', 'Las Tunas','Matanzas', 
                    'Pinar del Río', 'Sancti Spíritus', 'Santiago de Cuba','Villa Clara']
        
        self.Rainfall_Models = ['Gradient Booosting Regressor', 'Linear Regression', 'Support Vector Regressor']

        # Humidity Standarization
        self.crop_df = pd.read_csv('../Datasets/Crop_recommendation.csv')
        self.humidity_Series = self.crop_df['humidity']
        self.humidity_mean = round(self.humidity_Series.mean(), 3)
        self.humidity_median = round(self.humidity_Series.median(), 3)

        # Crop Models Init
        self.DT_Crop = load('../Models/Crop/Crop_DT.joblib')
        self.GNB_Crop =  load('../Models/Crop/Crop_GNB.joblib')
        self.RF_Crop =  load('../Models/Crop/Crop_RF.joblib')
        self.SVM_Crop =  load('../Models/Crop/Crop_SVM.joblib')
        self.WC_Crop =  load('../Models/Crop/Crop_WC.joblib')
        # List of Crop Models
        self.crop_models_list = [self.DT_Crop, self.GNB_Crop, self.RF_Crop, self.SVM_Crop, self.WC_Crop]
        self.crop_models_names = ['DecissionTree', 'GausianNaivyBayes', 'RandomForest', 'SupportVectorMachine', 'WeigthedClassifier']

        # Weather Models Init
        ## Temperature
        self.LR_Temp = load('../Models/Weather/Temp_LR.joblib')

        ## Rainfall
        self.GBR_Rain = load('../Models/Weather/Rainfall_GBR.joblib')
        self.KR_Rain = load('../Models/Weather/Rainfall_KR.joblib')
        self.LR_Rain = load('../Models/Weather/Rainfall_LR.joblib')
        self.SVR_Rain = load('../Models/Weather/Rainfall_SVR.joblib')
        # List of Crop Models
        self.rain_model_list = [self.GBR_Rain, self.LR_Rain, self.SVR_Rain]

    
    def Crop_Prediction(self,X, model = 4):
        prediction = self.crop_models_list[model].predict(X)
        return self.Crop_Label[prediction[0]]
    
    def Crop_Prediction_Each_Clf(self,X):
        
        # Getting Prediction From Each Classifier
        predictions = []
        for item in self.crop_models_list:
            pred = item.predict(X)
            predictions.append(pred[0])
        
        # Getting label
        predict_label = []
        for item in predictions:
            predict_label.append(self.Crop_Label[item])
        
        return predict_label
    
    def Temp_Prediction(self,X):
        prediction = self.LR_Temp.predict(X)
        return prediction
    
    def Rain_Prediction(self,X, model = 2):
        prediction = self.rain_model_list[model].predict(X)
        return prediction
    
    def Normalize_Weather_Models_Input(self, State, Year):
        input = [State, Year]
        x = np.array(input).reshape(1,2)
        return x
    
    def Normalize_Crop_Models_Input(self, temp, rain, humidity, ph, N, P, K):
        input = [temp, rain, humidity, ph, N, P, K]
        x = np.array(input).reshape(1,7)
        return x
    
    def Future_Crop_Prediction(self, State, Year, humidity, ph, N, P, K, MultiPrediction=True):
        if humidity == 0:
            '0==mean'
            humidity = self.humidity_mean
        elif humidity == 1:
            '1==median'
            humidity =self.humidity_median
        else:
            humidity = humidity

        input_weather = [State, Year]
        temp = self.Temp_Prediction(self.Normalize_Weather_Models_Input(State, Year))
        rain = self.Rain_Prediction(self.Normalize_Weather_Models_Input(State, Year))

        input_crop = [temp[0], rain[0], humidity, ph, N, P, K]
        
        if MultiPrediction: 
            crop = self.Crop_Prediction_Each_Clf(self.Normalize_Crop_Models_Input(temp[0], rain[0], humidity,ph, N, P, K))
        else:
            crop = self.Crop_Prediction(self.Normalize_Crop_Models_Input(temp[0], rain[0], humidity,ph, N, P, K))
        
        return crop

# cp = CropWeather_Models()


# final_result = cp.Future_Crop_Prediction(0,2100,0,50,30,100,80)
# print(final_result)
