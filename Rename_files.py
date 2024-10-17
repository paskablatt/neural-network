import os

def rename_images_in_folders(parent_folder_path):
    # Перебираем все папки в указанной родительской директории
    for subdir, _, files in os.walk(parent_folder_path):
        # Счетчик для переименования файлов
        count = 1
        for file in files:
            # Проверяем, является ли файл изображением
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # Формируем новый имя файла. Сохраняем оригинальное расширение.
                file_extension = os.path.splitext(file)[1]
                new_name = f"{count}{file_extension}"

                # Полные пути к файлу источник и новый файл
                source = os.path.join(subdir, file)
                destination = os.path.join(subdir, new_name)

                # Переименовываем файл
                os.rename(source, destination)
                print(f"Renamed '{source}' to '{destination}'")

                # Увеличиваем счетчик
                count += 1

parent_folder_path = "D:\\photo"

rename_images_in_folders(parent_folder_path)