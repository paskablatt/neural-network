import os

def rename_images_with_continuous_numbers(main_folder):
    current_number = 1  # Начальное число для имен файлов

    # Проходим по всем подпапкам в указанной папке
    for root, dirs, files in os.walk(main_folder):
        # Пропускаем основной каталог, работаем только с вложенными папками
        if root == main_folder:
            continue

        # Отсортируем файлы для упорядоченности
        files = sorted(files)

        # Шаг 1: Сначала временно переименовываем файлы, чтобы избежать конфликтов
        temp_names = []
        for idx, file_name in enumerate(files):
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                # Создаем временное имя
                temp_name = f"temp_{idx}{os.path.splitext(file_name)[1]}"
                temp_path = os.path.join(root, temp_name)
                os.rename(file_path, temp_path)
                temp_names.append(temp_path)

        # Шаг 2: Переименовываем файлы в окончательные имена с учетом текущего числа
        for temp_path in temp_names:
            # Получаем новое имя
            new_name = f"{current_number}{os.path.splitext(temp_path)[1]}"
            new_path = os.path.join(root, new_name)

            # Переименовываем файл
            os.rename(temp_path, new_path)
            print(f"Переименован: {temp_path} -> {new_path}")

            # Увеличиваем текущий номер
            current_number += 1

# Укажите путь к основной папке
main_folder_path = r"D:\1datasetLERA"  # Замените на ваш путь
rename_images_with_continuous_numbers(main_folder_path)
