import os
import shutil


# Функция для получения целевой папки по номеру папки
def get_target_folder(folder_number):
    if folder_number in range(1, 9) or folder_number in range(81, 89):
        return 'ENT'
    elif folder_number in range(9, 17) or folder_number in range(89, 97):
        return 'ISF'
    elif folder_number in range(49, 57) or folder_number in range(105, 113):
        return 'INF'
    elif folder_number in range(33, 41) or folder_number in range(113, 121):
        return 'ENF'
    elif folder_number in range(57, 65) or folder_number in range(97, 105):
        return 'EST'
    elif folder_number in range(17, 25) or folder_number in range(73, 81):
        return 'ESF'
    elif folder_number in range(25, 33) or folder_number in range(65, 73):
        return 'INT'
    elif folder_number in range(41, 49) or folder_number in range(121, 129):
        return 'IST'
    return None


# Путь к основной папке
root_folder = r"D:\фас проф"  # Укажите путь к основной папке
# Папка, куда нужно перемещать файлы
target_root_folder = r"D:\1FAS"  # Укажите путь к целевой папке

# Обходим папки от 1 до 128
for i in range(1, 129):
    folder_path = os.path.join(root_folder, str(i))
    if os.path.isdir(folder_path):
        fac_folder = os.path.join(folder_path, 'фас')
        if os.path.isdir(fac_folder):
            # Получаем целевую папку для перемещения изображений
            target_folder = get_target_folder(i)
            if target_folder:
                target_folder_path = os.path.join(target_root_folder, target_folder)
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)

                # Перемещаем все изображения из папки фас в целевую папку
                for filename in os.listdir(fac_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        src_file = os.path.join(fac_folder, filename)
                        dst_file = os.path.join(target_folder_path, filename)
                        shutil.move(src_file, dst_file)
                        print(f"Перемещено: {filename} в {target_folder_path}")
