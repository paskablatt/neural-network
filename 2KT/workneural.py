import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 98.0  # Минимальная уверенность для обработки изображения

# Загрузка модели
model = load_model("face_profile_classifier1234.h5")

def classify_and_organize_images(base_folder, output_folder):
    # Проверяем, существуют ли базовая и выходная папки
    if not os.path.exists(base_folder):
        print(f"Директория {base_folder} не найдена.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Проходим по всем папкам и файлам
    for root, dirs, files in os.walk(base_folder):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Проверяем, является ли текущий файл изображением
            try:
                img = image.load_img(file_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Получаем предсказание
                prediction = model.predict(img_array)[0][0]

                # Определяем класс и уверенность
                label = "Профиль" if prediction < 0.5 else "Фас"
                confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

                # Пропускаем изображения с низкой уверенностью
                if confidence < CONFIDENCE_THRESHOLD:
                    print(f"Пропущено (низкая уверенность): {file_path} ({confidence:.2f}%)")
                    continue

                # Создаём папки в выходной директории
                relative_folder_name = os.path.relpath(root, base_folder)
                dest_base_folder = os.path.join(output_folder, relative_folder_name)
                dest_label_folder = os.path.join(dest_base_folder, label)
                os.makedirs(dest_label_folder, exist_ok=True)

                # Копируем файл в соответствующую папку
                dest_path = os.path.join(dest_label_folder, filename)
                shutil.copy(file_path, dest_path)

                print(f"Скопировано: {file_path} -> {dest_path} ({confidence:.2f}%)")

            except Exception as e:
                print(f"Ошибка обработки файла {file_path}: {e}")

# Укажите путь к базовой папке с изображениями и выходной папке
base_folder_path = r"D:\1datasetLERA"
output_folder_path = r"D:\fasprofil"
classify_and_organize_images(base_folder_path, output_folder_path)
