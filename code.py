import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Charger les données
data = pd.read_excel('combined data.xlsx')

# Définir la colonne cible
target_column = 'corespandance'

# Prétraitement
X = data.drop(target_column, axis=1)
y = data[target_column]

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle : Arbre de décision
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Résultats
print("Arbre de décision - Matrice de confusion:\n", confusion_matrix(y_test, dt_predictions))
