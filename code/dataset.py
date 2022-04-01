import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_pima_indians_diabetes_dataset():
    """
    Loads the Pima Indians Diabetes Dataset.
    The dataset has to be pre-downloaded into `datasets/`.
    I.e. from here: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """

    df = pd.read_csv("datasets/diabetes.csv")

    # preprocessing of the dataset
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    fill_values = SimpleImputer(missing_values=0, strategy="mean", copy=False)

    df[feature_columns] = fill_values.fit_transform(df[feature_columns])

    X = df[feature_columns]
    y = df.Outcome

    return X, y

def load_stroke_prediction_dataset():
    """
    Loads the Stroke Prediction Dataset.
    The dataset has to be pre-downloaded into `datasets/`.
    I.e. from here: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
    """

    df = pd.read_csv('datasets/healthcare-dataset-stroke-data.csv')

    # preprocessing of the dataset
    # code taken from: https://www.kaggle.com/code/raghavtandon1305/performing-xg-boost
    df = df[['id', 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status',
            'hypertension', 'stroke', 'age', 'avg_glucose_level', 'bmi', 'heart_disease']]

    X = df.iloc[:, 1:11].values
    y = df.iloc[:, -1].values

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, [9]])
    X[:, [9]] = imputer.transform(X[:, [9]])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    return X, y