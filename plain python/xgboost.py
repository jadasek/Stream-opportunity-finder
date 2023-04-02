# %%
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# %%
data = pd.read_csv('OV2.csv', on_bad_lines='skip', sep=';')
data = pd.DataFrame(data)
data.info()

# %%
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M')
data['Hour'] = data['Date'].dt.hour
data['Minute'] = data['Date'].dt.hour
data['Weekday'] = data['Date'].dt.weekday
data['Moving_streams'] = data['Streams'].rolling(10, min_periods=1).mean()

data.head()

# %%
X = data[['Hour','Minute','Weekday']]
y = data[['Moving_streams']]

#y = data[['Moving_streams']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
y_test = y_test.reset_index()

# %%

xgb_model = xgb.XGBRegressor(n_estimators = 1000)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

#For measuring accuracy 
y_pred = y_pred.astype(int)
y_test['Moving_streams'] = y_test['Moving_streams'].astype(int)



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



# %%

new_data = pd.DataFrame({'Hour': [17], 'Minute': [0], 'Weekday': [0]})
prediction = xgb_model.predict(new_data)

print('Score: ', prediction)

# %%
xgb_model.feature_importances_

# %%
from sklearn.ensemble import IsolationForest

X = data[['Hour','Minute','Weekday','Streams']]

isof = IsolationForest(n_estimators=1000, contamination=0.1)
isof.fit(X)

new_data = pd.DataFrame({'Hour': [17], 'Minute': [12], 'Weekday': [0], 'Streams': [11]})
display(new_data)
prediction = isof.predict(new_data)

print('Wynik: ', prediction)


# %%
df_grouped = data.groupby(['Weekday', 'Hour'])['Moving_streams'].mean().reset_index()

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

weeks_name = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(35, 6))

for i in weeks_name.keys():
    group = df_grouped[df_grouped['Weekday'] == i]
    sns.lineplot(data=group, x='Hour', y='Moving_streams', ax=axs[i])
    axs[i].set_title(f"Avg amount of streams on {weeks_name[i]}")
    axs[i].set_xlabel("Hour")
    axs[i].set_ylabel("Streams")
    axs[i].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[i].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[i].set_xlim(left=0, right=23)
    axs[i].set_ylim(bottom=0, top=15)
    axs[i].grid()
    
plt.show()

# %%
df_grouped = data.groupby(['Hour'])['Moving_streams'].mean().reset_index()

plt.plot(df_grouped['Hour'], df_grouped['Moving_streams'])
plt.title("Avg amount of streams at specific hour")
plt.xlabel("Hour")
plt.ylabel("Streams")
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().set_xlim(left=0, right=23)
plt.grid()