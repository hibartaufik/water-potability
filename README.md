# Project 4: Machine Learning - Water Potability
![manu-schwendener-zFEY4DP4h6c-unsplash](https://user-images.githubusercontent.com/74480780/129901651-37ca0b7d-bf48-49c0-bd90-f97b11fc9377.jpg)

Photo by <a href="https://unsplash.com/@manuschwendener?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Manu Schwendener</a> on <a href="https://unsplash.com/s/photos/water-drink?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Overview
1. Membuat model Machine Learning dengan algoritma Random Forest
2. Dataset berasal dari kaggle.com dengan nama 'Water Quality', disusun oleh Aditya Kadiwal
3. Dataset memiliki 10 kolom
    - **'ph'**
    - **'Hardness'**
    - **'Solids'**
    - **'Chloramines'**
    - **'Sulfate'**
    - **'Conductivity'**
    - **'Organic_carbon'**
    - **'Trihalomethanes'**
    - **'Turbidity'**
    - **'Potability'**
   
   Untuk melihat keterangan setiap kolom, bisa di cek [disini](https://www.kaggle.com/adityakadiwal/water-potability)

4. Terdapat 6 tahapan dalam mengolah data dan membuat model, yaitu:
    - Import Libraries & Dataset
    - Exploratory Data Analysis
    - Data Preprocessing
    - Splitting & Modeling
    - Model Evaluation
    - Save Model
    
    Tahapan di atas bukan merupakan tahapan yang baku, tahapan dapat disesuaikan berdasarkan karakteristik data dan studi kasus
    
5. Project menggunakan dataset berasal kaggle, disusun oleh Aditya Kadiwal. Dapat diakses [disini](https://www.kaggle.com/adityakadiwal/water-potability)
 
## 1. Import Libraries & Dataset
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```

```
df = pd.read_csv('./data/water_potability.csv')
df.head()
```
![image](https://user-images.githubusercontent.com/74480780/130020276-5d2a3b14-1e72-49a3-b57b-5f3d8a1d1a7c.png)

## 2. Exploratory Data Analysis
## 3. Data Preprocessing
### 3.1 Handling Missing Value
### 3.2 Remove Outliers
### 3.3 Balancing Data Target
### 3.4 Feature Scaling
## 4. Splitting & Modeling
## 5. Model Evaluation
### 5.1 Hyperparameter Tuning dengan GridSeacrhCV
### 5.2 Confusion Matrix
## 6. Save Model
