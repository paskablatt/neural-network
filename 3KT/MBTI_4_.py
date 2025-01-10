import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score

# Загрузка модели FaceNet для извлечения эмбеддингов
facenet_model = tf.keras.models.load_model('facenet_keras_2024.h5')

def get_embedding(model, image):
    """
    Получение эмбеддинга из изображения с использованием модели FaceNet.
    """
    image = np.expand_dims(image, axis=0)
    embedding = model.predict(image, verbose=0)
    return embedding[0]

# Загрузка моделей классификации
model_FAS_E_I = load_model('FAS_E_I.h5')
model_FAS_N_S = load_model('FAS_N_S.h5')
model_FAS_T_F = load_model('FAS_T_F.h5')
model_PROF_J_P = load_model('PROF_J_P.h5')

def classify_image(model, image):
    """
    Классификация изображения (фас или профиль) с использованием модели.
    Возвращает предсказание и точность.
    """
    embedding = get_embedding(facenet_model, image)
    prediction = model.predict(np.expand_dims(embedding, axis=0))
    predicted_class = (prediction > 0.5).astype("int32")
    accuracy = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
    return predicted_class[0], accuracy

def predict_mbti(fas1_path, fas2_path, profile_path):
    """
    Определение типа личности MBTI на основе путей к изображениям фас и профиль.
    """
    # Загрузка изображений
    fas1_image = load_img(fas1_path, target_size=(160, 160))
    fas2_image = load_img(fas2_path, target_size=(160, 160))
    profile_image = load_img(profile_path, target_size=(160, 160))

    # Классификация для FAS_E_I
    fas1_class, fas1_accuracy = classify_image(model_FAS_E_I, fas1_image)
    fas2_class, fas2_accuracy = classify_image(model_FAS_E_I, fas2_image)
    fas_e_i_class = fas1_class if fas1_accuracy > fas2_accuracy else fas2_class
    fas_e_i_accuracy = fas1_accuracy if fas1_accuracy > fas2_accuracy else fas2_accuracy

    # Вывод результатов для FAS_E_I
    print(f"FAS_E_I - fas1: {('I' if fas1_class == 1 else 'E')} (accuracy: {fas1_accuracy:.2f}), fas2: {('I' if fas2_class == 1 else 'E')} (accuracy: {fas2_accuracy:.2f})")
    print(f"Лучший результат для FAS_E_I: {('I' if fas_e_i_class == 1 else 'E')} с точностью {fas_e_i_accuracy:.2f}")

    # Классификация для FAS_N_S
    fas1_class, fas1_accuracy = classify_image(model_FAS_N_S, fas1_image)
    fas2_class, fas2_accuracy = classify_image(model_FAS_N_S, fas2_image)
    fas_n_s_class = fas1_class if fas1_accuracy > fas2_accuracy else fas2_class
    fas_n_s_accuracy = fas1_accuracy if fas1_accuracy > fas2_accuracy else fas2_accuracy

    # Вывод результатов для FAS_N_S
    print(f"FAS_N_S - fas1: {('S' if fas1_class == 1 else 'N')} (accuracy: {fas1_accuracy:.2f}), fas2: {('S' if fas2_class == 1 else 'N')} (accuracy: {fas2_accuracy:.2f})")
    print(f"Лучший результат для FAS_N_S: {('S' if fas_n_s_class == 1 else 'N')} с точностью {fas_n_s_accuracy:.2f}")

    # Классификация для FAS_T_F
    fas1_class, fas1_accuracy = classify_image(model_FAS_T_F, fas1_image)
    fas2_class, fas2_accuracy = classify_image(model_FAS_T_F, fas2_image)
    fas_t_f_class = fas1_class if fas1_accuracy > fas2_accuracy else fas2_class
    fas_t_f_accuracy = fas1_accuracy if fas1_accuracy > fas2_accuracy else fas2_accuracy

    # Вывод результатов для FAS_T_F
    print(f"FAS_T_F - fas1: {('T' if fas1_class == 1 else 'F')} (accuracy: {fas1_accuracy:.2f}), fas2: {('T' if fas2_class == 1 else 'F')} (accuracy: {fas2_accuracy:.2f})")
    print(f"Лучший результат для FAS_T_F: {('T' if fas_t_f_class == 1 else 'F')} с точностью {fas_t_f_accuracy:.2f}")

    # Классификация для PROF_J_P
    profile_class, profile_accuracy = classify_image(model_PROF_J_P, profile_image)

    # Вывод результатов для PROF_J_P
    print(f"PROF_J_P - профиль: {('P' if profile_class == 1 else 'J')} (точность: {profile_accuracy:.2f})")

    # Формирование результата MBTI
    mbti_type = ""
    mbti_type += 'I' if fas_e_i_class == 1 else 'E'
    mbti_type += 'S' if fas_n_s_class == 1 else 'N'
    mbti_type += 'T' if fas_t_f_class == 1 else 'F'
    mbti_type += 'P' if profile_class == 1 else 'J'

    # Расчет финальной точности
    final_accuracy = fas_e_i_accuracy * fas_n_s_accuracy * fas_t_f_accuracy * profile_accuracy

    return mbti_type, final_accuracy

# Пример использования
fas1_path = r"D:\testvector8\1\I\73203.jpeg"  # Укажите путь к изображению фас1
fas2_path = r"D:\testvector8\1\I\73203.jpeg"  # Укажите путь к изображению фас2
profile_path = r"D:\popkorn\5\prof.jpeg"  # Укажите путь к изображению профиля
mbti_type, final_accuracy = predict_mbti(fas1_path, fas2_path, profile_path)

print(f"\nПредсказанный тип MBTI: {mbti_type}")
print(f"Финальная точность распознавания: {final_accuracy:.2f}")
