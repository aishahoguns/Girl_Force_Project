import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
class PredictPipeline:
    def __init__(self):
        self.pipeline = None
        self.model = RandomForestRegressor()
 

    def preprocess(self, data):
        num_cols = data.select_dtypes(['float64', 'int64']).columns
        cat_cols = data.columns.difference(num_cols)
        self.pipeline = Pipeline([('preprocessor', ColumnTransformer(
            transformers=[('num', StandardScaler(), num_cols),
                          ('cat', OneHotEncoder(), cat_cols)],
            remainder= 'passthrough'
           )
        )])
       
        return self.pipeline.fit_transform(data)

    def Train_Model(self, X, y):
        preprocessed_X = self.preprocess(X)
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, y, test_size=0.25, random_state=89)
        self.model.fit(X_train, y_train)


    calories_data = pd.read_csv('data/calories.csv')
    exercise_data = pd.read_csv('data/exercise.csv')

    

       
    def predict(self, instance):
        X = exercise_data.drop('User_ID', axis=1)
        y = calories_data['Calories']
        self.Train_Model(X,y)    
        
        instance_df = pd.DataFrame(instance)
        instance_transformed = self.pipeline.transform(instance_df)
        return self.model.predict(instance_transformed)
        



predict_pipeline = PredictPipeline()
calories_data = pd.read_csv('data/calories.csv')
exercise_data = pd.read_csv('data/exercise.csv')

X = exercise_data.drop('User_ID', axis=1)
y = calories_data['Calories']

predict_pipeline.Train_Model(X,y)





