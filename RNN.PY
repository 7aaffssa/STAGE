from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Simuler des données séquentielles
X_train_seq = np.random.rand(1000, 10, 1)  # 1000 séquences de longueur 10
y_train_seq = np.random.randint(2, size=1000)
X_test_seq = np.random.rand(200, 10, 1)  # 200 séquences
y_test_seq = np.random.randint(2, size=200)

# Création du modèle RNN
model_rnn = Sequential()
model_rnn.add(LSTM(50, input_shape=(10, 1)))
model_rnn.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model_rnn.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2)

# Prédictions
y_pred_rnn = (model_rnn.predict(X_test_seq) > 0.5).astype("int32")

# Rapport de classification
print("\nRapport de Classification RNN :")
print(classification_report(y_test_seq, y_pred_rnn))

# Matrice de confusion
print("Matrice de Confusion RNN :")
print(confusion_matrix(y_test_seq, y_pred_rnn))