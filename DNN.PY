import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix

# Charger le fichier Excel
df = pd.read_excel('combined data.xlsx', sheet_name='Sheet1')

# Nettoyage des données
df = df.dropna()
df_encoded = pd.get_dummies(df, columns=['Métamodèle 1', 'Métamodèle 2', 'Élément 1', 'Élément 2'])

# Sélection des caractéristiques et de la cible
X = df_encoded.drop('Correspondance', axis=1)
y = df_encoded['Correspondance']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle DNN
model_dnn = Sequential()
model_dnn.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model_dnn.add(Dense(32, activation='relu'))
model_dnn.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model_dnn.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Prédictions
y_pred_dnn = (model_dnn.predict(X_test_scaled) > 0.5).astype("int32")

# Rapport de classification
print("\nRapport de Classification DNN :")
print(classification_report(y_test, y_pred_dnn))

# Matrice de confusion
print("Matrice de Confusion DNN :")
print(confusion_matrix(y_test, y_pred_dnn))