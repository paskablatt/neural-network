import numpy as np
import requests
from PIL import Image
import tensorflow as tf
from io import BytesIO

from google.colab import files
uploaded = files.upload()

import tensorflow as tf

# Загрузка модели из файла
model = tf.keras.models.load_model('my_model.keras')

# Вывод структуры модели
model.summary()



# Замените URL на URL нового изображения
class_names = ['ENFJ', 'ENTP', 'ESFJ', 'INFP', 'INTJ', 'ISFP', 'ISTJ']
image_url = "https://shablon.klev.club/uploads/posts/2024-09/shablon-klev-club-7f6b-p-litso-cheloveka-1.jpg"

# Загрузка изображения без кэширования
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Пример, если модель ожидает размер входного изображения 128x128
img_height, img_width = 128, 128

# Изменение размера изображения
img = img.resize((img_width, img_height))

# Преобразование изображения в массив
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



# Замените путь на путь к вашему изображению на компьютере
image_path = "/content/1 (659)(45).jpeg"  # Убедитесь, что указано имя файла и расширение
class_names = ['ENFJ', 'ENTP', 'ESFJ', 'INFP', 'INTJ', 'ISFP', 'ISTJ']

# Загрузка изображения с компьютера
img1 = Image.open(image_path)

# Пример, если модель ожидает размер входного изображения 128x128
img_height, img_width = 128, 128

# Изменение размера изображения
img1 = img1.resize((img_width, img_height))

# Преобразование изображения в массив
img_array = tf.keras.utils.img_to_array(img1)
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
img1.show()