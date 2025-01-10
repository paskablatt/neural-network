import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Загрузка предобученной модели FaceNet
facenet_model = tf.keras.models.load_model('facenet_keras_2024.h5')

def get_embedding(model, image):
    """
    Получение эмбеддинга из изображения с использованием модели FaceNet.
    """
    image = np.expand_dims(image, axis=0)
    embedding = model.predict(image, verbose=0)
    return embedding[0]

def augment_images_and_labels(base_images, base_labels, datagen, augment_count=2):
    """
    Аугментация изображений и соответствующих меток.
    """
    augmented_images = []
    augmented_labels = []
    for image, label in zip(base_images, base_labels):
        image = np.expand_dims(image, axis=0)  # Добавляем измерение batch_size
        augmented_flow = datagen.flow(image, batch_size=1)  # Поток данных
        for _ in range(augment_count):
            augmented_image = next(augmented_flow)[0]
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)

# Загрузка исходных изображений
print("Загрузка данных...")
original_images = []
original_labels = []
dataset_path = r'D:\FAS_3_'  # Укажите путь к вашему датасету

for fold in os.listdir(dataset_path):
    fold_path = os.path.join(dataset_path, fold)
    if os.path.isdir(fold_path):
        for label in os.listdir(fold_path):
            label_path = os.path.join(fold_path, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    try:
                        image = load_img(image_path, target_size=(160, 160))
                        image = img_to_array(image) / 255.0  # Нормализация
                        original_images.append(image)
                        original_labels.append(label)
                    except Exception as e:
                        print(f"Ошибка обработки изображения {image_path}: {e}")

original_images = np.array(original_images)
original_labels = np.array(original_labels)

# Аугментация данных
print("Аугментация данных...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
augmented_images, augmented_labels = augment_images_and_labels(original_images, original_labels, datagen)

# Преобразование всех изображений в эмбеддинги
print("Создание эмбеддингов...")
augmented_embeddings = np.array([get_embedding(facenet_model, img) for img in augmented_images])

# Кодирование меток
encoder = LabelEncoder()
augmented_labels_encoded = encoder.fit_transform(augmented_labels)

# Настройка кросс-валидации
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_num = 1
all_fold_results = []


def build_classifier(num_classes):
    """Создание модели классификатора."""
    classifier = Sequential([
        Input(shape=(128,)),  # Входной слой для эмбеддингов размером 128
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Многоклассовая классификация
    ])
    classifier.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return classifier

# Выполнение кросс-валидации
num_classes = len(np.unique(augmented_labels_encoded))
for train_index, test_index in kfold.split(augmented_embeddings):
    print(f"Фолд {fold_num}:")

    # Разделение данных на тренировочные и тестовые
    X_train, X_test = augmented_embeddings[train_index], augmented_embeddings[test_index]
    y_train, y_test = augmented_labels_encoded[train_index], augmented_labels_encoded[test_index]

    # Создание модели
    classifier = build_classifier(num_classes)

    # Обучение модели
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = classifier.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    # Оценка модели на тестовых данных
    y_pred = np.argmax(classifier.predict(X_test), axis=1)
    fold_accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на фолде {fold_num}: {fold_accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    all_fold_results.append(fold_accuracy)
    fold_num += 1

# Итоговые результаты
mean_accuracy = np.mean(all_fold_results)
print(f"Средняя точность по всем фолдам: {mean_accuracy:.2f}")

# Сохранение финальной модели
final_classifier = build_classifier(num_classes)
final_classifier.fit(augmented_embeddings, augmented_labels_encoded, epochs=200, batch_size=128, verbose=0)
final_classifier.save('FAS8CLASS.h5')
print("Финальная модель сохранена как FAS8CLASS.h5")
