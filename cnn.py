from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Simuler des données d'images (64x64 pixels, 1 canal)
X_train_images = np.random.rand(1000, 64, 64, 1)  # 1000 images
X_test_images = np.random.rand(200, 64, 64, 1)  # 200 images
y_train = np.random.randint(2, size=1000)  # 1000 étiquettes
y_test = np.random.randint(2, size=200)  # 200 étiquettes

# Création du modèle CNN
model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Flatten())
model_cnn.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model_cnn.fit(X_train_images, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Évaluation
loss, accuracy = model_cnn.evaluate(X_test_images, y_test)

# Prédictions
y_pred_cnn = (model_cnn.predict(X_test_images) > 0.5).astype("int32")

# Rapport de classification
print("\nRapport de Classification CNN :")
print(classification_report(y_test, y_pred_cnn))

# Matrice de confusion
print("Matrice de Confusion CNN :")
print(confusion_matrix(y_test, y_pred_cnn))