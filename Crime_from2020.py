#DOCUMENTAZIONE DELLA FONTE
#il dataset scelto è recuperato da Kaggle, contiene i report dei crimini commessi a Los Angeles dal 2022 al Dicembre 2023 definendone l'area, la data, i dati della vittima, 
#il modus operandi dell'assassino e l'arma che è stata utilizzata.

#importazione di tutte le libreire necessarie
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

#salvataggio del dataset
df = pd.read_csv(r'Crime_from2020.csv')
df.head()

#stampa delle prime informazioni come: tipi di dato delle colonne 
df.info()
#e gli eventuali valori nulli
df.isnull().sum()


# Ridimensionamento del df in base alle esigenze 
colonne_da_mantenere = ["date_reported", "area", "area_name", "crime_code", "crime_description", "victim_age", "premise_code", "premise_description", "weapon_code", "weapon_description"]
df2 = df[colonne_da_mantenere].copy()
df2.to_csv("df2-file.csv", index=False)  
#eliminazione dei valori nulli per analisi dei dati migliore
df2 = df2.dropna()
#modifica dei tipi di dati per renderli più consoni
df2['date_reported'] = pd.to_datetime(df2['date_reported'])
df2['area_name'] = df2['area_name'].astype(str)
df2['crime_description'] = df2['crime_description'].astype(str)
df2['premise_description'] = df2['premise_description'].astype(str)
# Riempimento valori mancanti in premise_description
df2['premise_description'].fillna('', inplace=True)
#in weapon_code e weapon_description
df2['weapon_code'].fillna(0, inplace=True)
df2['weapon_description'].fillna('', inplace=True)

df2
df2.info()
df2.isnull().sum()


# OBBIETTIVO_1 --------------------------------
#analizzare i dati per trovare un "canone" di crimini, ovvero dove avvengono più spesso, in quale periodo ecc

#-- CRIMINI PER AREA 
sns.countplot (x="area_name", data=df2, order=df2["area_name"].value_counts ().index)
plt.xticks (rotation=90)
plt.show ()


#-- CRIMINI PER MESE E ANNO DI SEGNLAZIONE 
# Estrazione del mese e dell'anno dalla colonna 'date_reported'
df2['month_year'] = df2['date_reported'].dt.to_period('M')
# Calcolo delle frequenze dei mesi e anni di segnalazione e ordinamento 
month_year_counts = df2['month_year'].value_counts().sort_index()
# Countplot basato sulla data di segnalazione
plt.figure(figsize=(12, 6))
ax = sns.countplot(x="month_year", data=df2, order=month_year_counts.index, palette="viridis")
# Rotazione delle etichette sull'asse x per una migliore leggibilità
plt.xticks(rotation=45, ha="right")
# Aggiunta delle etichette dei valori sopra le barre
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=8, color='darkblue', xytext=(0, 5),
                textcoords='offset points')
plt.show()


#-- CRIMINI PER ETA' DELLE VITTIME
import numpy as np
import matplotlib.pyplot as plt
# Filtraggio dell'età delle vittime in modo che siano comprese tra 1 e 100
filtered_df = df2[(df2['victim_age'] > 1) & (df2['victim_age'] < 100)]
# Calcolo delle frequenze delle età nel DataFrame filtrato
age_counts = filtered_df['victim_age'].value_counts()
# Estrazione  dei valori e le relative frequenze
x_axes = age_counts.index
y_axes = age_counts.values
# Plot
sizes = np.random.uniform(15, 80, len(x_axes))
colors = np.random.uniform(15, 80, len(x_axes))
plt.scatter(x_axes, y_axes, s=sizes, c=colors, vmin=0, vmax=100)
plt.xlabel('victim_age')
plt.ylabel('count')
plt.show()


#-- CRIMINI PER TIPO
# Calcolo delle frequenze dei crime_code nel DataFrame filtrato
crime_counts = df2['crime_code'].value_counts()
# Estrazione dei valori e relative frequenze
x_axes = crime_counts.index
y_axes = crime_counts.values
# Plot
sizes = np.random.uniform(15, 80, len(x_axes))
colors = np.random.uniform(15, 80, len(x_axes))
plt.scatter(x_axes, y_axes, s=sizes, c=colors, vmin=0, vmax=100)
plt.xlabel('crime_code')
plt.ylabel('count')
plt.show()
#mi faccio restituire il crime code esatto
crime_counts = df2['crime_code'].value_counts()
crime_code_max_count = crime_counts.idxmax()
print(f"Il crime_code con il conteggio massimo è: {crime_code_max_count}")
#cerco la corrispondenza del crime_code al suo rispettivo crime_description
crime_code_624_description = df2[df2['crime_code'] == 624]['crime_description'].iloc[0]
print(f"crime_description for crime_code 624: {crime_code_624_description}")


#-- CRIMINI PER ARMA
# Filtraggio del DataFrame escludendo i weapon code uguali a zero
filtered_df = df2[df2['weapon_code'] != 0]
# Calcolo delle frequenze dei weapon code nel DataFrame filtrato
weapon_counts = filtered_df['weapon_code'].value_counts()
# Estrazione dei valori e relative frequenze
x_axes = weapon_counts.index
y_axes = weapon_counts.values
# Plot
sizes = np.random.uniform(15, 80, len(x_axes))
colors = np.random.uniform(15, 80, len(x_axes))
plt.scatter(x_axes, y_axes, s=sizes, c=colors, vmin=0, vmax=100)
plt.xlabel('weapon_code')
plt.ylabel('count')
plt.show()
filtered_df = df2[df2['weapon_code'] != 0]
weapon_counts = filtered_df['weapon_code'].value_counts()
weapon_code_max_count = weapon_counts.idxmax()
print(f"Il weapon_code con il conteggio massimo (escludendo 0) è: {weapon_code_max_count}")
weapon_code_400_description = df2[df2['weapon_code'] == 400]['weapon_description'].iloc[0]
print(f"weapon_description for weapon_code 400: {weapon_code_400_description}")


#-- CRIMINI PER LUOGO
# Calcolo delle frequenze dei crime_code nel DataFrame filtrato
premise_counts = df2['premise_code'].value_counts()
# Estrazione dei valori e relative frequenze
x_axes = premise_counts.index
y_axes = premise_counts.values
# Plot
sizes = np.random.uniform(15, 80, len(x_axes))
colors = np.random.uniform(15, 80, len(x_axes))
plt.scatter(x_axes, y_axes, s=sizes, c=colors, vmin=0, vmax=100)
plt.xlabel('premise_code')
plt.ylabel('count')
plt.show()
premise_counts = df2['premise_code'].value_counts()
premise_code_max_count = premise_counts.idxmax()
print(f"Il premise_code con il conteggio massimo è: {premise_code_max_count}")
premise_code_101_description = df2[df2['premise_code'] == 101.0]['premise_description'].iloc[0]
print(f"premise_description for premise_code 101: {premise_code_101_description}")


#Conclusione
#I CRIMINI SONO AVVENUTI PRINCIPALMENTE:
#a 77th Street di Los Angeles,
#nel Maggio del 2022,
#le vittime hanno età tra i 20 e 40 anni, 
#tipo di crimine: assalto 
#mezzo per crimine: violenza fisica 
#luogo del crimine: per strada 


#OBBIETTIVO_2 --------------------------------
#trovare un'eventuale correlazione tra i dati e analizzarla

# Esclusione colonne non numeriche
numeric_columns = df2.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df2[numeric_columns].corr()
# heatmap utilizzando seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Matrice di Correlazione')
plt.show()
#la correlazione tra crime_code e weapon_code è molto alta, quindi si andrà ad analizzare

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


#-- REGRESSIONE LOGISTICA, weapon_code
# X contiene le colonne features
X = df2[['weapon_code']]
# y contiene la colonna target (esito)
y = df2['outcome']
# dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardizzazione delle features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# modello di regressione logistica
model = LogisticRegression()
# Addestramento del modello
model.fit(X_train, y_train)
# previsioni sul set di test
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
accuracy =accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}".format(accuracy))


#-- REGRESSIONE LOGISTICA, crime_code
# X contiene le colonne features
X = df2[['crime_code']]
y = df2['outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
# previsioni sul set di test
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
accuracy =accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}".format(accuracy))