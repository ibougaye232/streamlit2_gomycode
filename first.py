import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
import joblib
open("C:/fii_db.csv")
fii_db=pd.read_csv("C:/fii_db.csv")


encoder=LabelEncoder()
for i in fii_db.columns:
  if fii_db[i].dtype==object:
    fii_db[i]=encoder.fit_transform(fii_db[i])
    
# Supprimer les colonnes 'year' et 'uniqueid' de fii_db
x = fii_db[["age_of_respondent","household_size","country","year","education_level","gender_of_respondent"]]
y = fii_db['bank_account']
sampler = SMOTEENN(sampling_strategy=0.6, random_state=42)
x_resample, y_resample = sampler.fit_resample(x, y)
sampler2=RandomUnderSampler(sampling_strategy=0.7,random_state=42)
x_resampled, y_resampled=sampler2.fit_resample(x_resample, y_resample)

# Diviser les données en ensemble d'entraînement et ensemble de test
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=20)

# Créer et entraîner le modèle RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,class_weight='balanced')

model.fit(x_train, y_train)

joblib.dump(model, 'trained_classifier_model.joblib')
print("finished")