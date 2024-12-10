import os
import shutil


# Функция для получения целевой папки по номеру папки
def get_target_folder_profile(folder_number):
    if (1 <= folder_number <= 16) or (49 <= folder_number <= 80) or (113 <= folder_number <= 128):
        return 'P'
    elif (17 <= folder_number <= 48) or (81 <= folder_number <= 112):
        return 'J'
    return None


# Путь к основной папке
root_folder = r"D:\фас проф"  # Укажите путь к основной папке
# Папка, куда нужно перемещать файлы
target_root_folder = r"D:\1PROF"  # Укажите путь к целевой папке

# Обходим папки от 1 до 128
for i in range(1, 129):
    folder_path = os.path.join(root_folder, str(i))
    if os.path.isdir(folder_path):
        profile_folder = os.path.join(folder_path, 'профиль')
        if os.path.isdir(profile_folder):
            # Получаем целевую папку для перемещения изображений
            target_folder = get_target_folder_profile(i)
            if target_folder:
                target_folder_path = os.path.join(target_root_folder, target_folder)
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)

                # Перемещаем все изображения из папки профиль в целевую папку
                for filename in os.listdir(profile_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        src_file = os.path.join(profile_folder, filename)
                        dst_file = os.path.join(target_folder_path, filename)
                        shutil.move(src_file, dst_file)
                        print(f"Перемещено: {filename} в {target_folder_path}")
