# %%
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import numpy as np
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#Modules for ML
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import classification_report,confusion_matrix

# %%
data = pd.read_csv('OV2.csv', on_bad_lines='skip', sep=';', )
data = pd.DataFrame(data)
data.info()

# %%
X = data.iloc[:, [10,11,12]]
y = data.iloc[:, 13]

#y = pd.to_numeric(y)
#y = [s.replace(',', '') for s in y]
y = y.replace(',', '.', regex=True)

y= np.asarray(y).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %%
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.fit(X_train, y_train, epochs=100)

# %%
def predict(data):
    pred = model.predict(data).flatten()
    pred = np.rint(pred)
    pred = pred.tolist()
    print(pred)
    print(y_test)
    return pred

y_pred_test = predict(X_test)

model.evaluate(X_test, y_test)

# %%
y_test