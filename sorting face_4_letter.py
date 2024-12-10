import os
import shutil


# Функция для получения целевой папки для категории E
def get_target_folder_E(folder_number):
    if (1 <= folder_number <= 8) or (17 <= folder_number <= 24) or (33 <= folder_number <= 40) or \
            (57 <= folder_number <= 64) or (73 <= folder_number <= 88) or (97 <= folder_number <= 104) or \
            (113 <= folder_number <= 120):
        return 'E'
    else:
        return 'I'


# Функция для получения целевой папки для категории N
def get_target_folder_N(folder_number):
    if (1 <= folder_number <= 8) or (25 <= folder_number <= 40) or (81 <= folder_number <= 88) or \
            (105 <= folder_number <= 120) or (49 <= folder_number <= 56) or (65 <= folder_number <= 72):
        return 'N'
    else:
        return 'S'


# Функция для получения целевой папки для категории T
def get_target_folder_T(folder_number):
    if (1 <= folder_number <= 8) or (25 <= folder_number <= 32) or (41 <= folder_number <= 48) or \
            (57 <= folder_number <= 72) or (81 <= folder_number <= 88) or (97 <= folder_number <= 104) or \
            (121 <= folder_number <= 128):
        return 'T'
    else:
        return 'F'


# Путь к основной папке
root_folder = r"D:\fasprofil"  # Укажите путь к основной папке
# Папка, куда нужно копировать файлы
target_root_folder = "D:\FAS EINSTF"  # Укажите путь к целевой папке

# Обходим папки от 1 до 128
for i in range(1, 129):
    folder_path = os.path.join(root_folder, str(i))
    if os.path.isdir(folder_path):
        fac_folder = os.path.join(folder_path, 'фас')
        if os.path.isdir(fac_folder):
            # Получаем целевую папку для копирования изображений в E/I, N/S, T/F
            target_folder_E = get_target_folder_E(i)
            target_folder_N = get_target_folder_N(i)
            target_folder_T = get_target_folder_T(i)

            # Копируем изображения в папки E/I, N/S и T/F
            for filename in os.listdir(fac_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    # Копирование изображений в папку E/I
                    target_folder_E_path = os.path.join(target_root_folder, target_folder_E)
                    if not os.path.exists(target_folder_E_path):
                        os.makedirs(target_folder_E_path)
                    src_file = os.path.join(fac_folder, filename)
                    dst_file_E = os.path.join(target_folder_E_path, filename)
                    shutil.copy(src_file, dst_file_E)

                    # Копирование изображений в папку N/S
                    target_folder_N_path = os.path.join(target_root_folder, target_folder_N)
                    if not os.path.exists(target_folder_N_path):
                        os.makedirs(target_folder_N_path)
                    dst_file_N = os.path.join(target_folder_N_path, filename)
                    shutil.copy(src_file, dst_file_N)

                    # Копирование изображений в папку T/F
                    target_folder_T_path = os.path.join(target_root_folder, target_folder_T)
                    if not os.path.exists(target_folder_T_path):
                        os.makedirs(target_folder_T_path)
                    dst_file_T = os.path.join(target_folder_T_path, filename)
                    shutil.copy(src_file, dst_file_T)

                    print(
                        f"Копировано: {filename} в {target_folder_E_path}, {target_folder_N_path}, {target_folder_T_path}")
