import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

data = pd.read_csv('OV2.csv', on_bad_lines='skip', sep=';')
data = pd.DataFrame(data)
data.info()

X = data[['Hour','Minute','Weekday']]
#y = data[['Streams','Dominant_streams','Viewers', 'Dominant_ratio']]

y = data[['Moving_streams']]
#y['Moving_viewers'] = y['Moving_viewers'].str.replace(',', '.').astype(float)
y['Moving_streams'] = y['Moving_streams'].str.replace(',', '.').astype(float)
y = y.drop(y.index[0:10])
X = X.drop(X.index[0:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
y_test = y_test.reset_index()


xgb_model = xgb.XGBRegressor(n_estimators = 1000)

#xgb_model.fit(X, y, eval_set=[(X, y), (X, y)],verbose=100)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

y_pred = y_pred.astype(int)
y_test['Moving_streams'] = y_test['Moving_streams'].astype(int)
#print(y_pred)
#print(y_test['Moving_streams'])


wynik = 0
wynik_list =[]
#For bigger games
# for i in range(len(y_test)):
#     temp = max(y_pred[i],y_test['Moving_streams'][i]) / min(y_pred[i],y_test['Moving_streams'][i])
#     temp = ((temp * 100) - 100) * 10
#     temp = 100 - temp
#     if temp > 0:
#         wynik += temp/100
#         wynik_list.append(temp)
#     else:
#         wynik += 0
#         wynik_list.append(0)

for i in range(len(y_test)):
    if y_test['Moving_streams'][i] == y_pred[i]:
        wynik += 1
        wynik_list.append(100)
    elif y_pred[i] == y_test['Moving_streams'][i] + 1 or  y_pred[i] == y_test['Moving_streams'][i] - 1:
        wynik += 0.5
        wynik_list.append(50)
    else:
        wynik += 0 
        wynik_list.append(0)

# For bigger games
# df = pd.DataFrame(y_pred)
# df['should'] = y_test['Moving_streams']
# df['result'] = wynik_list
# df.columns = ['is','should','result']
# display(df)

print(wynik/len(y_test))

df = pd.DataFrame(y_pred)
df['should'] = y_test['Moving_streams']
df['result'] = wynik_list
df.columns = ['is','should','result']
display(df)




new_data = pd.DataFrame({'Hour': [17], 'Minute': [0], 'Weekday': [4]})
prediction = xgb_model.predict(new_data)

print('Score: ', prediction)

xgb_model.feature_importances_

from sklearn.ensemble import IsolationForest

X = data[['Hour','Minute','Weekday','Streams']]

isof = IsolationForest(n_estimators=500)
isof.fit(X)

new_data = pd.DataFrame({'Hour': [9], 'Minute': [12], 'Weekday': [2], 'Streams': [5]})
prediction = isof.predict(new_data)

print('Wynik: ', prediction)
