import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# Установка параметров
img_width, img_height = 256, 256  # Размеры изображений
batch_size = 32  # Размер батча
epochs = 10  # Количество эпох

# Создание генераторов для загрузки изображений с аугментацией данных
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Измените путь к директории на fac_TF
train_generator = train_datagen.flow_from_directory(
    'fac_TF/train',  # Путь к обучающему набору данных
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Две категории: T и F
)

validation_generator = val_datagen.flow_from_directory(
    'fac_TF/val',  # Путь к валидационному набору данных
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Использование предобученной модели VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Замораживаем слои базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Создание новой модели
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout для регуляризации
model.add(layers.Dense(1, activation='sigmoid'))  # Используем сигмоид для бинарной классификации

# Компиляция модели с адаптивным методом оптимизации Adam
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Уменьшенная скорость обучения
              metrics=['accuracy'])

# Обучение модели с использованием обратного вызова EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Сохранение модели
model.save('mbti_tf_vgg16_model.h5')

# Визуализация результатов обучения
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Вызов функции для визуализации
plot_training_history(history)
