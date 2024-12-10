import os
import shutil
from sklearn.model_selection import train_test_split

# Укажите путь к вашей папке MBTI16
source_dir = 'MBTI16'
train_dir = os.path.join(source_dir, 'train')
val_dir = os.path.join(source_dir, 'val')

# Создаем папки для тренировочных и валидационных данных
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Проходим по всем папкам-классам
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if os.path.isdir(class_path):  # Проверяем, что это папка
        # Получаем список всех файлов в папке класса
        all_files = os.listdir(class_path)
        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

        # Создаем папки для класса в train и val
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Копируем файлы в соответствующие папки
        for file_name in train_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(train_dir, class_name))
        for file_name in val_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(val_dir, class_name))

print("Данные успешно разделены!")
