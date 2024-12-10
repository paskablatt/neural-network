import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Загрузка моделей
face_profile_model = load_model('face_profile_classifier1234.h5')
mbti_model = load_model('mbti_personality_classifier.h5')
mmpi_model = load_model('MMPI_personality_classifier.h5')

# Список названий классов MBTI
mbti_classes = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP',
                'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP',
                'ISFJ', 'ISFP', 'ISTJ', 'ISTP']

# Список названий классов MMPI (примерные названия, замените на актуальные)
mmpi_classes = [
    "депрессивный",  # 0
    "истерический",  # 1
    "компульсивный",  # 2
    "мазохистический",  # 3
    "нарциссический",  # 4
    "параноидальный",  # 5
    "психопатический",  # 6
    "шизоиздный",  # 7
]


# Функция для загрузки изображений и предсказания
def predict_personality(image_path1, image_path2):
    # Обработка первого изображения (фас)
    img1 = image.load_img(image_path1, target_size=(128, 128))
    img_array1 = image.img_to_array(img1) / 255.0
    img_array1 = np.expand_dims(img_array1, axis=0)

    # Предсказание фас или профиля для первого изображения
    face_profile_prediction1 = face_profile_model.predict(img_array1)
    face_profile_class1 = 'Фас' if face_profile_prediction1[0][0] > 0.5 else 'Профиль'

    face_profile_accuracy1 = face_profile_prediction1[0][0] if face_profile_class1 == 'Фас' else 1 - \
                                                                                                 face_profile_prediction1[
                                                                                                     0][0]

    print(
        f"Результат первой модели для первого изображения: {face_profile_class1} (Точность: {face_profile_accuracy1:.2f})")

    # Обработка для MBTI для первого изображения
    mbti_img1 = image.load_img(image_path1, target_size=(224, 224))
    mbti_array1 = image.img_to_array(mbti_img1) / 255.0
    mbti_array1 = np.expand_dims(mbti_array1, axis=0)

    mbti_prediction1 = mbti_model.predict(mbti_array1)
    mbti_class_index1 = np.argmax(mbti_prediction1)

    if face_profile_class1 == 'Фас':
        mbti_two_letters1 = f"_{mbti_classes[mbti_class_index1][1]}{mbti_classes[mbti_class_index1][2]}_"
        mbti_full_type1 = mbti_classes[mbti_class_index1]
        print(
            f"Результат второй модели (MBTI) для первого изображения: {mbti_two_letters1} (Полный тип: {mbti_full_type1}) (Точность: {mbti_prediction1[0][mbti_class_index1]:.2f})")

        # Предсказание для MMPI
        mmpi_img = image.load_img(image_path1, target_size=(224, 224))
        mmpi_array = image.img_to_array(mmpi_img) / 255.0
        mmpi_array = np.expand_dims(mmpi_array, axis=0)

        mmpi_prediction = mmpi_model.predict(mmpi_array)
        mmpi_class_index = np.argmax(mmpi_prediction)
        mmpi_class_name = mmpi_classes[mmpi_class_index]

        print(
            f"Результат третьей модели (MMPI): класс '{mmpi_class_name}' (Точность: {mmpi_prediction[0][mmpi_class_index]:.2f})")

    else:
        mbti_two_letters1 = f"{mbti_classes[mbti_class_index1][0]}__{mbti_classes[mbti_class_index1][3]}"
        mbti_full_type1 = mbti_classes[mbti_class_index1]
        print(
            f"Результат второй модели (MBTI) для первого изображения: {mbti_two_letters1} (Полный тип: {mbti_full_type1}) (Точность: {mbti_prediction1[0][mbti_class_index1]:.2f})")

    # Обработка второго изображения (противоположное)
    img2 = image.load_img(image_path2, target_size=(128, 128))
    img_array2 = image.img_to_array(img2) / 255.0
    img_array2 = np.expand_dims(img_array2, axis=0)

    # Предсказание фас или профиля для второго изображения
    face_profile_prediction2 = face_profile_model.predict(img_array2)
    face_profile_class2 = 'Фас' if face_profile_prediction2[0][0] > 0.5 else 'Профиль'

    face_profile_accuracy2 = face_profile_prediction2[0][0] if face_profile_class2 == 'Фас' else 1 - \
                                                                                                 face_profile_prediction2[
                                                                                                     0][0]

    print(
        f"Результат первой модели для второго изображения: {face_profile_class2} (Точность: {face_profile_accuracy2:.2f})")

    # Проверка на совпадение типов
    if (face_profile_class1 == face_profile_class2):
        print("Невозможно определить весь тип личности MBTI.")
        return

    # Обработка для MBTI для второго изображения
    mbti_img2 = image.load_img(image_path2, target_size=(224, 224))
    mbti_array2 = image.img_to_array(mbti_img2) / 255.0
    mbti_array2 = np.expand_dims(mbti_array2, axis=0)

    mbti_prediction2 = mbti_model.predict(mbti_array2)
    mbti_class_index2 = np.argmax(mbti_prediction2)

    if face_profile_class2 == 'Фас':
        mbti_two_letters2 = f"_{mbti_classes[mbti_class_index2][1]}{mbti_classes[mbti_class_index2][3]}"
        full_type_02 = mbti_classes[mbti_class_index2]
        print(
            f"Результат второй модели (MBTI) для второго изображения: {mbti_two_letters2} (Полный тип: {full_type_02}) (Точность: {mbti_prediction2[0][mbti_class_index2]:.02f})")

    else:
        two_letters_mbtyi_02 = f"{mbti_classes[mbti_class_index2][0]}__{mbti_classes[mbti_class_index2][3]}"
        full_type_02 = mbti_classes[mbti_class_index2]
        print(
            f"Результат второй модели (MBTI) для второго изображения: {two_letters_mbtyi_02} (Полный тип: {full_type_02}) (Точность: {mbti_prediction2[0][mbti_class_index2]:.02f})")
    full_mbti_type = f"{mbti_classes[mbti_class_index2][0]}{mbti_classes[mbti_class_index1][1]}{mbti_classes[mbti_class_index1][2]}{mbti_classes[mbti_class_index2][3]}"
    print(f"Полный тип личности и характера MBTI и MMPI: {full_mbti_type}, {mmpi_class_name}")

# Пример использования функции с двумя изображениями
predict_personality('fac_x.jpg', 'pro_y.jpg')