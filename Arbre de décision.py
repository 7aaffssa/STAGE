import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

file_path = r'C:\Users\USER\Desktop\stage\combined data.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier '{file_path}' est introuvable. Vérifiez le chemin et réessayez.")
data = pd.read_excel(file_path, sheet_name='Sheet1')
if 'Correspondance' not in data.columns:
    raise KeyError("La colonne 'Correspondance' est introuvable dans les données. Assurez-vous que le fichier est correct.")
data = data.dropna(subset=['Correspondance'])

# Sélectionner les colonnes pertinentes
X = data.drop(columns=['Correspondance'])  # Features
y = data['Correspondance']                # Cible

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n")
print(report)
fictitious_results = pd.DataFrame({
    'Real': y_test.values[:10] if len(y_test) >= 10 else y_test.values,
    'Predicted': y_pred[:10] if len(y_pred) >= 10 else y_pred
})
print("\nExemples de résultats fictifs:\n")
print(fictitious_results)