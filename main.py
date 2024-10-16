import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling

from google.colab import files
uploaded = files.upload()


!unzip photo_class.zip -d /content/dataset

!ls /content/dataset/photo_class


#Задание параметров

batch_size = 32
img_width = 128
img_height = 128

#Загрузка тренировочного и валидационного датасетов
#Этот метод загружает изображения из директории, организованной по подпапкам
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/dataset/photo_class', #Путь
    validation_split=0.2,  #Указывает, что 20% данных будут выделены для валидационной выборки, а 80% — для тренировочной.
    subset="training",  #Эти параметры указывают, какую часть данных загружать — тренировочную или валидационную
    seed=123,  # Фиксированное значение для воспроизводимого разделения данных на тренировочную и валидационную выборки
    image_size=(img_height, img_width), #Указывает размер
    batch_size=batch_size)  #Указывает, сколько изображений будет загружаться за один раз

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/dataset/photo_class',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names #Возвращает список классов
print(f"Class names: {class_names}")

#Оптимизация загрузки данных
# cache
#Оптимизирует подачу данных, используя кэширование, перемешивание и предварительную загрузку,
#что помогает ускорить обучение модели и улучшить её точность.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


from tensorflow.keras import layers, Sequential

# create model
num_classes = len(class_names)
model = Sequential([
    # Нормализация
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # Аугментация данных
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),

    # Слои свёрточной нейросети
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Регуляризация
    layers.Dropout(0.2),

    # Полносвязные слои
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# print model summary
model.summary()



# train the model
epochs = 10 # количество эпох тренировки
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs)

# visualize training and validation results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



# train the model
epochs = 20 # количество эпох тренировки
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs)

# visualize training and validation results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Замените URL на URL нового изображения
image_url = "https://i.pinimg.com/736x/b0/15/8d/b0158de4599ed378ed0a07a018a03229.jpg"
image_path = tf.keras.utils.get_file('my_new_image', origin=image_url)
# Пример, если модель ожидает размер входного изображения 128x128
img_height, img_width = 128, 128

# Загрузка изображения с изменением размера
img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Добавляем размер batch

# Предсказания модели
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Вывод результата
print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
    class_names[np.argmax(score)],
    100 * np.max(score)
))

# Отображение изображения
img.show()
