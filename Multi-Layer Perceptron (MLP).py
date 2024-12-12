import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

# Vérifier si le fichier existe
file_path = r'C:\Users\USER\Desktop\stage\combined data.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable. Vérifiez le chemin et réessayez.")

# Charger les données
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Vérifier si la colonne 'Correspondance' existe
if 'Correspondance' not in data.columns:
    raise KeyError("La colonne 'Correspondance' est introuvable dans les données. Assurez-vous que le fichier est correct.")

# Supprimer les lignes contenant des valeurs NaN dans la colonne cible
data = data.dropna(subset=['Correspondance'])

# Sélectionner les colonnes pertinentes
X = data.drop(columns=['Correspondance'])  # Features
y = data['Correspondance']                # Cible

# Encodage des colonnes non numériques si nécessaire
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle MLP
model = Sequential()

# Ajouter des couches cachées
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Prédire et évaluer
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Afficher les métriques
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n")
print(report)

# Générer des résultats fictifs
fictitious_results = pd.DataFrame({
    'Real': y_test.values[:10] if len(y_test) >= 10 else y_test.values,
    'Predicted': y_pred[:10] if len(y_pred) >= 10 else y_pred
})
print("\nExemples de résultats fictifs:\n")
print(fictitious_results)
