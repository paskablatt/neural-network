import os
from PIL import Image

def convert_and_resize_images(input_folder):
    # Проверяем, существует ли папка
    if not os.path.exists(input_folder):
        print(f"Директория {input_folder} не найдена.")
        return

    # Проходимся по всем файлам в папке и её подкаталогах
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            # Получаем полный путь к файлу
            file_path = os.path.join(root, filename)

            # Проверяем, является ли текущий файл изображением
            if os.path.isfile(file_path):
                try:
                    # Открываем изображение
                    with Image.open(file_path) as img:
                        width, height = img.size

                        # Обрезаем изображение, чтобы сделать его квадратным
                        if width < height:
                            # Обрезать по высоте
                            new_height = width
                            top = (height - new_height) // 2
                            img = img.crop((0, top, width, top + new_height))
                        elif height < width:
                            # Обрезать по ширине
                            new_width = height
                            left = (width - new_width) // 2
                            img = img.crop((left, 0, left + new_width, height))

                        # Изменяем размер изображения до 256x256 пикселей
                        img = img.resize((160, 160))

                        # Удаляем старое расширение у имени файла
                        basename = os.path.splitext(filename)[0]

                        # Создаём новое имя файла с расширением .jpeg
                        new_filename = f"{basename}.jpeg"
                        new_file_path = os.path.join(root, new_filename)

                        # Сохраняем изображение в формате JPEG
                        img.convert("RGB").save(new_file_path, "JPEG")

                        # Удаляем старый файл, если имя изменилось
                        if file_path != new_file_path:
                            os.remove(file_path)

                        print(f"Сохранено: {new_file_path}")
                except Exception as e:
                    print(f"Не удалось обработать файл {file_path}: {e}")

# Запуск функции
convert_and_resize_images(r"D:\1datasetLERA")
