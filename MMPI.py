import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Параметры
image_size = (224, 224)
batch_size = 32

# Генератор данных
datagen = ImageDataGenerator(
    rescale=1.0/255.0  # Оставляем нормализацию
)

train_generator = datagen.flow_from_directory(
    'MMPI/train',  # Путь к тренировочной папке
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    'MMPI/val',  # Путь к валидационной папке
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = models.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',  # Следим за валидационными потерями
    patience=10,         # Останавливаем обучение, если за 10 эпох нет улучшения
    restore_best_weights=True  # Восстанавливаем лучшие веса
)

# Обучение модели с увеличением числа эпох
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,          # Увеличиваем максимальное количество эпох
    callbacks=[early_stopping]  # Добавляем EarlyStopping
)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# График потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.save('MMPI_personality_classifier.h5')

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy:.2f}")
