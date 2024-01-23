import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r'Crime_from2020.csv')
df.head()
df.info()
df.isnull().sum()

# OBBIETTIVO_1
df2 = df[["weapon_code", "area", "victim_sex", "victim_descent", "date_occurred"]]

df2["weapon_code"] = df2["weapon_code"].astype(float)
df2["area"] = df2["area"].factorize()[0]
df2["victim_sex"] = df2["victim_sex"].factorize()[0]
df2["victim_descent"] = df2["victim_descent"].factorize()[0]

corr = df2.corr()

sns.heatmap(corr, annot=True, cmap="RdBu")
plt.show()


X = df2.drop(columns=['outcome']).to_numpy()
y = df2['outcome'].to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

df2['area'].plot(kind='hist', edgecolor='black')

clf = LogisticRegression()
clf.fit(X_train, y_train)

#---------------------------------------------------------------
#previsione di un attacco futuro
# Ad esempio, si può usare un modello lineare per stimare la data di un futuro crimine in base all'area
X = df2.drop(columns=['outcome']).to_numpy()
y = df2['outcome'].to_numpy()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
df2['area'].plot(kind='hist', edgecolor='black')
clf = LogisticRegression()
clf.fit(X_train, y_train)
#--------------------
#aggiungiamo la colonna outcome che vale 0 se è presente il crime_code_1 altrimenti 1
df2.loc[:, "outcome"] = df2.loc[:, "crime_code_1"].apply(lambda x: 1 if np.isnan(x) else 0)
#-------------------------------------------------
#tolgo le colonne che ritengo meno rilevanti
df_corr = df.drop(['date_occurred', 'area_name', 'crime_description', 'modus_operandi', 'victim_descent', 'premise_code', 'victim_sex', 'premise_description', 'weapon_description', 'status_description', 'crime_code_2', 'crime_code_3', 'crime_code_4', 'location', 'cross_street'], axis=1)
df_corr['date_reported'] = pd.to_datetime(df_corr['date_reported'])
df_corr["status"] = df_corr["status"].factorize()[0]
df_corr["outcome"] = df_corr.isnull().all(axis=1)
#---------------------------------------------

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = GeoDataFrame(df, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)



from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326"
)

world = gpd.read_file(get_path(df[df['area_name']=='Central']))

# We restrict to South America.
ax = world.clip([-90, -55, -25, 15]).plot(color="white", edgecolor="black")

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color="red")

plt.show()
cols_to_drop = ['division_number', 'date_reported', 'date_occurred', 'area', 'reporting_district', 'part', 'crime_code', 'crime_description', 'modus_operandi', 'status', 'status_description', 'crime_code_1', 'crime_code_2', 'crime_code_3', 'crime_code_4', 'location', 'cross_street']
# uso il metodo drop per eliminare le colonne
df.drop(cols_to_drop, axis=1, inplace=True)

#----------------------------------
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from geodatasets import get_path

# assumo che il tuo dataframe si chiami df
# creo df_map con solo le colonne area_name, longitude e latitude
df_map = df.loc[:, ['area_name', 'longitude', 'latitude']]

# creo un geodataframe con i dati di latitudine e longitudine
gdf = geopandas.GeoDataFrame(df_map, geometry=geopandas.points_from_xy(df_map.longitude, df_map.latitude), crs="EPSG:4326")

# leggo il file della mappa del mondo
world = geopandas.read_file(get_path("naturalearth.land"))

# restringo la mappa a Los Angeles
ax = world.clip([-118.6682, 33.7037, -118.1553, 34.3373]).plot(color="white", edgecolor="black")
# plotto il geodataframe sulla mappa con i colori per area e la legenda
gdf.plot(ax=ax, hue='area_name', legend=True)

plt.show()

#----------
import pandas as pd
import geopandas
import folium
# assumo che il tuo dataframe si chiami df
# creo df_map con solo le colonne area_name, longitude e latitude
df_map = df.loc[:, ['area_name', 'longitude', 'latitude']]

# creo un geodataframe con i dati di latitudine e longitudine
gdf = geopandas.GeoDataFrame(df_map, geometry=geopandas.points_from_xy(df_map.longitude, df_map.latitude), crs="EPSG:4326")

# creo un oggetto mappa di folium, centrato su Los Angeles
m = folium.Map(location=[34.0522, -118.2437], zoom_start=10)

# aggiungo il geodataframe alla mappa
folium.GeoJson(gdf).add_to(m)

# visualizzo la mappa
m

#-------------------------------------------------
#modifichiamo i tipi di dati per renderli più consoni
df['date_reported'] = pd.to_datetime(df['date_reported'])
df['date_occurred'] = pd.to_datetime(df['date_occurred'])

# Seleziona le colonne categoriche da codificare
categorical_columns = ["area_name", "crime_description", "modus_operandi", "victim_sex", "victim_descent", "premise_description", "weapon_description", "status", "status_description"]

# Esegui l'encoding one-hot per le colonne categoriche
df_encoded = pd.get_dummies(df, columns=categorical_columns)

df_encoded = pd.get_dummies(df2, columns=['area_name'])


#REGRESSIONE LOGISTICA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Supponendo che 'status' sia la tua variabile di destinazione binaria
X = df2[['crime_code', 'weapon_code', ...]]  # Inserisci le tue variabili predittive
y = df2['status']

# Dividi il dataset in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea e addestra il modello di regressione logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Esegui previsioni sul set di test
y_pred = model.predict(X_test)

# Valuta le prestazioni del modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, y_pred))






df2['count'] = df2['weapon_code'].map(weapon_counts)
new_columns = ['weapon_code', 'count', 'outcome']
logistic_regression = df2[new_columns].copy()
logistic_regression
sns.scatterplot(data=logistic_regression, x='weapon_code', y='count', hue='outcome')
X = logistic_regression.drop(columns=['outcome'])
y = logistic_regression['outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#outcome
#outcome 1 se c'è almeno un dato frequnte frequente, altrimenti 0
df2['outcome'] = np.where((df2['crime_code'] == 624) | ((df2['weapon_code'] == 400) & (df2['weapon_code'] != 0)), 1, 0)
#cambiare tipo di dato di outcome da categorico a intero
df2["outcome"] = df2["outcome"].astype(int)
df2.info()
# Esclusione colonne non numeriche
numeric_columns = df2.select_dtypes(include=['float64', 'int64', 'int32']).columns
correlation_matrix = df2[numeric_columns].corr()

# heatmap utilizzando seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')

plt.title('Matrice di Correlazione')
plt.show()
sns.countplot(data=df2, x='outcome', )


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix