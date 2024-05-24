__all__ = ['ImgEmbeddingKnn']

from functools import singledispatch
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import ResNet50, VGG16  # Add more models as needed
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet_v3
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ImgEmbeddingKnn:
    supported_models = {
        'MobileNetV3Small': (MobileNetV3Small, preprocess_mobilenet_v3),
        'MobileNetV3Large': (MobileNetV3Large, preprocess_mobilenet_v3),
        'ResNet50': (ResNet50, preprocess_resnet50),
        'VGG16': (VGG16, preprocess_vgg16),
    }
    tf_model_params = dict(weights='imagenet', include_top=False, pooling='avg')

    def __init__(self, base_model: str | tuple[object, callable]='MobileNetV3Small', n_neighbors: int=5):
        """Create model that firstly calculate image embeddings and then classify embedding with KNN

        Args:
            base_model (str | tuple[object, callable], optional): If string provided corresponding pretrained tensorflow model will be loaded.
                                                                  If you want use custom model provide model, and preprocessing function.
                                                                  Defaults to 'MobileNetV3Small'.
            n_neighbors (int, optional): sklearn knn parameter. Defaults to 5.
        """
        self.n_neighbors = n_neighbors
        self.base_model, self.preprocess_input = self._initialize_base_model(base_model)
        self.base_model.trainable = False
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.embeddings = None
        self.labels = None

    def _initialize_base_model(self, base_model):
        if isinstance(base_model, str):
            model_cls, preprocess_input = self.supported_models.get(base_model, (None, None))
            if model_cls == None:
                raise ValueError(f"Unsupported model name: {base_model}, supported names are: {tuple(self.supported_models.keys())}")
            model = model_cls(**self.tf_model_params)
        else:
            model, preprocess_input = base_model

        return model, preprocess_input

    # def load_and_preprocess_image(self, img_path):
    def load_and_preprocess_image(self, img: str | np.ndarray):
        if isinstance(img, str):
            img = image.load_img(img)#, target_size=(224, 224))
            img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)
        return img

    def get_image_embedding(self, img: str | np.ndarray):
        img_array = self.load_and_preprocess_image(img)
        embedding = self.base_model.predict(img_array, verbose=0)
        return embedding.flatten()

    def train(self, img_paths, labels, validation_size: None | float = 0.2, random_state=0xCAFFE):
        print("Collecting embeddings:")
        self.labels = labels
        self.embeddings = self.__collect_embeddings(img_paths)
        self.__train_knn(self.embeddings, labels, validation_size, random_state)

    def __collect_embeddings(self, img_paths):
        embeddings = []
        for img_path in tqdm(img_paths):
            embeddings.append(self.get_image_embedding(img_path))
        embeddings = np.array(embeddings)
        return embeddings

    def __train_knn(self, embeddings, labels, validation_size, split_random_state):
        if validation_size is None:
            self.__train_knn_without_validator(embeddings, labels)
        else:
            self.__train_knn_with_validator(embeddings, labels, validation_size, split_random_state)

    def __train_knn_without_validator(self, embeddings, labels):
        X_train, y_train = embeddings, labels
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        print(f"Training complete. Train accuracy: {train_accuracy:.2f}.")

    def __train_knn_with_validator(self, embeddings, labels, validation_size: float , split_random_state):
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels,
            test_size=validation_size,
            random_state=split_random_state)
        self.knn.fit(X_train, y_train)
        y_val_pred = self.knn.predict(X_val)
        y_train_pred = self.knn.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Training complete. Train accuracy: {train_accuracy:.2f}. Validation accuracy: {val_accuracy:.2f}")

    def predict(self, img_path):
        new_embedding = self.get_image_embedding(img_path)
        prediction = self.knn.predict([new_embedding])
        return prediction[0]

    def save_model(self, file_path):
        model_data = {
            'knn': self.knn,
            'base_model': self.base_model,
            'n_neighbors': self.n_neighbors
        }
        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path):
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
        instance = cls(base_model=model_data['base_model'], n_neighbors=model_data['n_neighbors'])
        instance.knn = model_data['knn']
        return instance
