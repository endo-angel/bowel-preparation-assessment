
import os
import numpy as np
import time

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.xception import preprocess_input as xception_preprocess_input

from utils import load_fold_data, split_train_valid_fold


def prepare_data(train_data_path, valid_data_path, target_img_size, class_count, bs, preprocess_input):
    X, y, file_list = load_fold_data(train_data_path, target_img_size)
    X = np.asarray(X)
    y = np.asarray(y) if class_count == 2 else to_categorical(y, class_count)

    X_valid, y_valid, _ = load_fold_data(valid_data_path, target_img_size)

    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid) if class_count == 2 else to_categorical(y_valid, class_count)

    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0,
        rotation_range=20,
        zoom_range=0.1,
        fill_mode='constant',
        vertical_flip=True,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    valid_datagen = ImageDataGenerator(
        fill_mode='constant',
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow(X, y, batch_size=bs)
    valid_generator = valid_datagen.flow(X_valid, y_valid, batch_size=bs)

    return train_generator, valid_generator


def get_optimizer(optimizer_type, lr=0.001):
    if optimizer_type == 'SGD':
        return SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer_type == 'RMSprop':
        return optimizers.RMSprop(lr=lr)
    elif optimizer_type == 'AdamOptimizer':
        return tf.train.AdamOptimizer(lr)
    else:
        print('Error optimizer type:', optimizer_type)


def create_model(input_img_size, model_type, class_count, opti_type, lr=0.001, unfreeze_layers=None):
    input_tensor = Input((input_img_size[0], input_img_size[1], 3))
    model_set_list = [(1, 'sigmoid', 'binary_crossentropy'), (class_count, 'softmax', 'categorical_crossentropy')]
    setting = model_set_list[0] if class_count == 2 else model_set_list[1]
    units, activation, loss = setting
    optimizers = get_optimizer(opti_type, lr)

    print('create %s model, img size: %s, %s, optimizer: %s, lr: %s, unfreeze_layers: %s'
          % (model_type, input_img_size, loss, opti_type, lr, unfreeze_layers))

    if model_type == 'vgg':
        base_model = VGG16(input_tensor=input_tensor, weights='imagenet')
        fc2 = base_model.get_layer('fc2').output

        prediction = Dense(units, activation=activation, name='last_dense')(fc2)
        model = Model(base_model.input, prediction)
        for layer in model.layers[:-3]:
            layer.trainable = False

    elif model_type == 'resnet':
        base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        for layers in base_model.layers:
            layers.trainable = False
        if unfreeze_layers:
            for layers in base_model.layers[-unfreeze_layers:]:
                layers.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        prediction = Dense(units, activation=activation, name='last_dense')(x)
        model = Model(base_model.input, prediction)

    elif model_type == 'densenet':
        base_model = DenseNet169(input_tensor=input_tensor, weights='imagenet', include_top=False)

        for layers in base_model.layers:
            layers.trainable = False
        if unfreeze_layers:
            for layers in base_model.layers[-unfreeze_layers:]:
                layers.trainable = True

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.3)(x)
        x = Flatten()(x)

        prediction = Dense(units, activation=activation, name='last_dense')(x)
        model = Model(base_model.input, prediction)

    elif model_type == 'xception':
        base_model = Xception(weights='imagenet', include_top=False)
        for layers in base_model.layers:
            layers.trainable = False
        if unfreeze_layers:
            for layer in base_model.layers[-unfreeze_layers:]:
                layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(units, activation=activation)(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    else:
        print('create model Error: model type error: ', model_type)
        return None

    model.compile(optimizer=optimizers, loss=loss, metrics=['accuracy'])
    return model


def fit_gen(model, model_id, weights_path, class_count, train_data, valid_data, steps, valid_steps, epochs=20):
    class_mode = 'binary' if class_count == 2 else 'categorical'
    print("train from imgs model_id=%s ,class_mode=%s" % (model_id, class_mode))

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    weight_file = weights_path + '/weights' + str(model_id) + '.hdf5'
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    check_point = ModelCheckpoint(weight_file, monitor='val_loss', save_best_only=True, save_weights_only=True)

    callbacks_list = [early_stop, check_point]

    history = model.fit_generator(
        train_data,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=valid_data,
        validation_steps=valid_steps,
        callbacks=callbacks_list)

    return history


def get_train_curve(title, x_label, y_label, y1, y1_label, y2, y2_label, save_file):
    plt.figure(figsize=(7, 7))
    plt.subplot(1, 1, 1)

    plt.plot(y1, label=y1_label)
    plt.plot(y2, label=y2_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    y_min = min(plt.ylim())
    y_max = max(plt.ylim())
    y_diff = y_max - y_min
    y_label_min = max(0, y_min - y_diff*0.2)
    y_label_max = y_max + y_diff*0.2
    plt.ylim([y_label_min, y_label_max])

    plt.legend(loc='lower right')
    plt.title(title)

    if save_file:
        plt.savefig(save_file)


def get_train_history(history, save_path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    time_stamp = time.strftime("%y%m%d_%H%M%S_", time.localtime(time.time()))

    save_file = os.path.join(save_path, time_stamp+"acc.png") if save_path else None
    get_train_curve('Training and Validation Accuracy',
                    'epoch',
                    'Accuracy',
                    acc, 'Training Accuracy',
                    val_acc, 'Validation Accuracy',
                    save_file)

    save_file = os.path.join(save_path, time_stamp+"loss.png") if save_path else None
    get_train_curve('Training and Validation Loss',
                    'epoch',
                    'Cross Entropy',
                    loss, 'Training Loss',
                    val_loss, 'Validation Loss',
                    save_file)


def save_model(model, model_id, weights_path, save_weight=True):
    model_json = model.to_json()
    json_file = weights_path + '/model' + str(model_id) + '.json'
    with open(json_file, "w") as json_file_handle:
        json_file_handle.write(model_json)

    if save_weight:
        weights_file = weights_path + '/weights' + str(model_id) + '.hdf5'
        try:
            model.save_weights(weights_file, save_format='h5')
        except:
            model.save_weights(weights_file)


# configs
root_path = r'F:\测试'
all_img_path = root_path + '/res'
train_path = root_path + '/res_train'
valid_path = root_path + '/res_test'
test_path = root_path + '/test'
train_valid_split_ratio = 0.2

weights_path = root_path + '/cache'


validate_split_ratio = 0.2
img_size = (224, 224)

model_name = 'resnet'
model_id = 1
opti_type = 'RMSprop'
lr = 1e-4
unfreeze_layers = 25

epochs = 10
batch_size = 64


#
if __name__ == '__main__':
    if not os.path.exists(train_path):
        split_train_valid_fold(all_img_path, train_path, valid_path, train_valid_split_ratio)

    class_count = len(os.listdir(train_path))
    preprocess_input = None
    if model_name == 'densenet':
        preprocess_input = densenet_preprocess_input
    elif model_name == 'xception':
        preprocess_input = xception_preprocess_input

    train_data, valid_data = prepare_data(train_path, valid_path, img_size, class_count, batch_size, preprocess_input)
    model = create_model(img_size, model_name, class_count, opti_type, lr, unfreeze_layers)

    K.set_learning_phase(1)
    steps = train_data.n // batch_size
    valid_steps = valid_data.n // batch_size
    history = fit_gen(model, model_id, weights_path, class_count, train_data, valid_data, steps, valid_steps, epochs)

    get_train_history(history, weights_path)

    save_model(model, model_id, weights_path)


