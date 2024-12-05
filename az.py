import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

# Charger le fichier Excel
df = pd.read_excel('combined data.xlsx')

# Prétraitement des données
# Encoder la colonne "Correspondance" (1 et 0)
le = LabelEncoder()
df['Correspondance'] = le.fit_transform(df['Correspondance'])

# Sélectionner les caractéristiques (features) et la cible (target)
X = df[['Métamodèle 1', 'Élément 1', 'Description Élément 1', 'Métamodèle 2', 'Élément 2', 'Description Élément 2']]
y = df['Correspondance']

# Convertir les caractéristiques en variables numériques (vous pouvez utiliser le TF-IDF ou d'autres méthodes)
X = X.apply(lambda x: x.astype(str).str.cat(sep=' '), axis=1)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de Machine Learning
# Vectorisation des textes
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train_vec, y_train)

# Évaluation du modèle
y_pred_rf = rf.predict(X_test_vec)
print("Random Forest Classifier Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Modèle de Deep Learning
# Créer un modèle simple
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_vec.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Pour une classification binaire
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train_vec.toarray(), y_train, epochs=10, batch_size=32)

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test_vec.toarray(), y_test)
print(f"Deep Learning Model Accuracy: {accuracy}")