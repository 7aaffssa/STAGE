import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# Charger le fichier Excel
df = pd.read_excel('combined_data.xlsx', sheet_name='Sheet1')

# Nettoyage des données
df = df.dropna()  # Supprimer les lignes avec des valeurs manquantes

# Encodage des colonnes catégorielles
df_encoded = pd.get_dummies(df, columns=['Métamodèle 1', 'Métamodèle 2', 'Élément 1', 'Élément 2'])

# Encodage de la colonne cible
label_encoder = LabelEncoder()
df_encoded['Correspondance'] = label_encoder.fit_transform(df_encoded['Correspondance'])

# Sélection des caractéristiques et de la cible
X = df_encoded.drop('Correspondance', axis=1)  # Caractéristiques
y = df_encoded['Correspondance'].values  # Cible

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction pour évaluer les modèles
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*50 + "\n")

# 1. Régression Logistique
print("Régression Logistique:")
evaluate_model(LogisticRegression(max_iter=200), X_test, y_test)

# 2. Support Vector Machine (SVM)
print("Support Vector Machine:")
evaluate_model(SVC(), X_test, y_test)

# 3. Forêts Aléatoires
print("Forêts Aléatoires:")
evaluate_model(RandomForestClassifier(), X_test, y_test)

# 4. Arbre de Décision
print("Arbre de Décision:")
evaluate_model(DecisionTreeClassifier(), X_test, y_test)

# ==========================
# Algorithmes de Deep Learning avec PyTorch
# ==========================

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test)

# 1. Réseau de Neurones Artificiels (ANN)
class ANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Entraîner et évaluer le modèle ANN
ann_model = ANN(X_train.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

# Entraînement
for epoch in range(10):
    ann_model.train()
    optimizer.zero_grad()
    outputs = ann_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Réseau de Neurones Artificiels (ANN):")
evaluate_model(ann_model, X_test_tensor, y_test_tensor)

# 2. Réseau de Neurones Convolutifs (CNN)
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * ((input_size - 2) // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Ajouter une dimension pour le canal
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Aplatir
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Entraîner et évaluer le modèle CNN
cnn_model = CNN(X_train.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Entraînement
for epoch in range(10):
    cnn_model.train()
    optimizer.zero_grad()
    outputs = cnn_model(X_train_tensor.unsqueeze(1))  # Ajouter dimension pour le canal
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Réseau de Neurones Convolutifs (CNN):")
evaluate_model(cnn_model, X_test_tensor.unsqueeze(1), y_test_tensor)

# 3. Réseau de Neurones Profonds (DNN)
# Utilise le même modèle que l'ANN, donc pas besoin de redéfinir

print("Réseau de Neurones Profonds (DNN):")
evaluate_model(ann_model, X_test_tensor, y_test_tensor)

# 4. Réseau de Neurones Récurrents (RNN)
class RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(1))  # Ajouter une dimension pour le batch
        x = x[:, -1, :]  # Prendre la dernière sortie
        return self.fc(x)

# Entraîner et évaluer le modèle RNN
rnn_model = RNN(X_train.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

# Entraînement
for epoch in range(10):
    rnn_model.train()
    optimizer.zero_grad()
    outputs = rnn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Réseau de Neurones Récurrents (RNN):")
evaluate_model(rnn_model, X_test_tensor, y_test_tensor)