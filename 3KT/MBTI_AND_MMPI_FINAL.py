import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Загрузка предобученных моделей
facenet_model = load_model('facenet_keras_2024.h5')
fas8_model = load_model('FAS8CLASS.h5')
prof_j_p_model = load_model('PROF_J_P.h5')
mmpi_model = load_model('MMPI_personality_classifier2.h5')  # Загрузка модели для классификации MMPI

# Классы для MMPI
mmpi_classes = [
    "депрессивный", "истерический", "компульсивный", "мазохистический",
    "нарциссический", "параноидальный", "психопатический", "шизоиздный"
]

def get_embedding(model, image):
    """
    Получение эмбеддинга из изображения с использованием модели FaceNet.
    """
    image = np.expand_dims(image, axis=0)
    embedding = model.predict(image, verbose=0)
    return embedding[0]

def preprocess_image(image_path):
    """
    Загрузка и предобработка изображения для моделей.
    """
    image = load_img(image_path, target_size=(160, 160))
    image = img_to_array(image) / 255.0  # Нормализация
    return image

def classify_fas_images(fas_model, embeddings):
    """
    Классификация изображений фас с использованием модели FAS8CLASS.
    """
    predictions = fas_model.predict(embeddings)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_probabilities = np.max(predictions, axis=1)
    return predicted_classes, predicted_probabilities

def classify_profile_image(profile_model, embedding):
    """
    Классификация профильного изображения с использованием модели PROF_J_P.
    """
    prediction = profile_model.predict(np.expand_dims(embedding, axis=0))[0][0]
    predicted_class = 'P' if prediction > 0.5 else 'J'  # Меняем местами J и P
    probability = prediction if prediction > 0.5 else 1 - prediction
    return predicted_class, probability

def classify_mmpi_personality(mmpi_model, image):
    """
    Классификация изображения для определения типа личности с использованием MMPI.
    """
    image = np.expand_dims(image, axis=0)
    predictions = mmpi_model.predict(image)
    predicted_class = np.argmax(predictions)
    predicted_probability = np.max(predictions)
    return mmpi_classes[predicted_class], predicted_probability

# Пути к изображениям
fas1_path = r"D:\popkorn\5\fas1.jpeg"  # Укажите путь к изображению фас1
fas2_path = r"D:\popkorn\5\fas2.jpeg"  # Укажите путь к изображению фас2
profile_path = r"D:\popkorn\5\prof.jpeg"  # Укажите путь к изображению профиля

# Предобработка изображений
fas1_image = preprocess_image(fas1_path)
fas2_image = preprocess_image(fas2_path)
profile_image = preprocess_image(profile_path)

# Создание эмбеддингов
fas1_embedding = get_embedding(facenet_model, fas1_image)
fas2_embedding = get_embedding(facenet_model, fas2_image)
profile_embedding = get_embedding(facenet_model, profile_image)

# Классификация фас изображений
fas_classes = ['ENF', 'ENT', 'ESF', 'EST', 'INF', 'INT', 'ISF', 'IST']
fas_embeddings = np.array([fas1_embedding, fas2_embedding])
fas_predicted_classes, fas_probabilities = classify_fas_images(fas8_model, fas_embeddings)

# Выбор лучшего класса фас
if fas_probabilities[0] > fas_probabilities[1]:
    best_fas_class = fas_classes[fas_predicted_classes[0]]
    best_fas_probability = fas_probabilities[0]
else:
    best_fas_class = fas_classes[fas_predicted_classes[1]]
    best_fas_probability = fas_probabilities[1]

# Классификация профильного изображения
profile_class, profile_probability = classify_profile_image(prof_j_p_model, profile_embedding)

# Классификация личностного типа MMPI
mmpi_personality, mmpi_probability = classify_mmpi_personality(mmpi_model, fas1_image)  # Используем фас1 для MMPI

# Вычисление финальной точности
final_accuracy = best_fas_probability * profile_probability

# Определение типа MBTI
mbti_type = best_fas_class + profile_class

# Вывод результатов
print(f"Классификация фас1: {fas_classes[fas_predicted_classes[0]]} с точностью {fas_probabilities[0]:.2f}")
print(f"Классификация фас2: {fas_classes[fas_predicted_classes[1]]} с точностью {fas_probabilities[1]:.2f}")
print(f"Лучший класс фас: {best_fas_class} с точностью {best_fas_probability:.2f}")
print(f"Классификация профиля: {profile_class} с точностью {profile_probability:.2f}")
print(f"Классификация MMPI: {mmpi_personality} с точностью {mmpi_probability:.2f}")
print(f"Тип MBTI: {mbti_type}")
print(f"Финальная точность MBTI: {final_accuracy:.2f}")
