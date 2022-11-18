# sudo apt-get update
# sudo apt install graphviz -y

import os, argparse, gc
import pickle, joblib
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model #, get_custom_objects
from tensorflow.keras.optimizers import SGD, Nadam, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import sigmoid
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A

def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history[0])
    plt.plot(train_history[1])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history[2])
    plt.plot(train_history[3])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.close(fig)
    return fig
def optimizer_f(optimizer):
    if optimizer == "sgd":
        return SGD(learning_rate=0.001, momentum=0.9, decay=0, nesterov=True)
    if optimizer == "nadam":
        return Nadam(learning_rate=0.001)
    if optimizer == 'adam':
        return Adam(learning_rate=0.001)
#def swish(x, beta = 1):
#    return (x * sigmoid(beta * x))
#get_custom_objects().update({'swish': swish})
def transforms(p=0.75):
    return A.Compose([
        A.RandomToneCurve(scale=0.3, p=1),
        #A.CLAHE(p=0.5),
        #A.ColorJitter(brightness=1,contrast=0, saturation=0, hue=0),
        #A.RandomBrightnessContrast(),
        A.RandomGamma(p=1),
        A.Blur(blur_limit=3, p=1)
    ], p=p)
def augmentor(image):
    aug = transforms()
    aug_img = aug(image = np.array(image).astype('uint8'))['image']
    aug_img = preprocess_input(aug_img)
    return aug_img
def callbacks_list_fn(file, loss, save_weights_only_, earlystop = True):
    if loss=="sparse":
        monitor_acc = 'val_sparse_categorical_accuracy'
    else:
        monitor_acc = 'val_accuracy'
        
    lrate = ReduceLROnPlateau(monitor = monitor_acc, factor=0.75, patience=10, mode='max', min_delta=0.00001, 
                              cooldown=0, threshold_mode='rel', min_lr=1e-8)    
    ck_callback_acc = ModelCheckpoint(file+'/weights_acc.h5', monitor=monitor_acc, mode='max', 
                                       save_best_only=True, save_weights_only=save_weights_only_)
    ck_callback_los = ModelCheckpoint(file+'/weights_loss.h5', monitor='val_loss', mode='min', 
                                      save_best_only=True, save_weights_only=save_weights_only_)
    csv_callback = CSVLogger(file+'/log.csv', append=True, separator=',')
    
    if earlystop:
        earlyst = EarlyStopping(monitor="val_loss", patience = 40, mode='min', min_delta=0.00001)    
        return [ck_callback_acc, ck_callback_los, csv_callback, lrate, earlyst]
    else:
        return [ck_callback_acc, ck_callback_los, csv_callback, lrate]

def model_build(activation_, normal_, l2_, optimizer_, pooling_, image_type, loss_):
    # 預先訓練好的模型 -- Xception_dataset, 不含後三層(辨識層)
    # base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
    # base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
    # base_model.trainable = False
    if image_type=='merge':
        base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
    if 'chanel3' in image_type:
        base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
    if image_type=='chanel6':
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 6))
    if image_type=='chanel12':
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 12))
    if image_type=='multi':
        base_model_1 = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
        base_model_2 = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
        # 連接自訂層
        for layer in base_model_1.layers:
            layer._name = layer._name + str("_1")
        for w in base_model_1.weights:
            w._handle_name = w.name + str("_1")
        for layer in base_model_2.layers:
            layer._name = 'EP_' + layer._name + str("_2")
        for w in base_model_2.weights:
            w._handle_name = 'EP_' + w.name + str("_2")
        base_model = layers.concatenate([base_model_1.output, base_model_2.output])
    
    if pooling_ == "max":
        if image_type=='multi':
            x = layers.GlobalMaxPooling2D()(base_model)
        else:
            x = layers.GlobalMaxPooling2D()(base_model.output)
    if pooling_ == "ave":
        if image_type=='multi':
            x = layers.GlobalAveragePooling2D()(base_model)
        else:
            x = layers.GlobalAveragePooling2D()(base_model.output)
    if pooling_ == "conv":
        if image_type=='multi':
            x = layers.Conv2D(1280, (7, 7), activation="relu")(base_model)
        else:
            x = layers.Conv2D(1280, (7, 7), activation="relu")(base_model.output)
    if pooling_ == 'all':
        x1 = layers.GlobalMaxPooling2D()(base_model.output)
        x2 = layers.GlobalAveragePooling2D()(base_model.output)
        x3 = layers.Conv2D(1280, (7, 7), activation="relu")(base_model.output)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Activation(activation_)(x3)
        x3 = layers.Flatten()(x3)
        x = layers.concatenate([x1, x2, x3])
    
    if pooling_ == "conv":
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_)(x)
        x = layers.Flatten()(x)
    else:        
        if normal_ == 'drop':
            x = layers.Dropout(0.25)(x)
        if normal_ == 'batch':
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation_)(x)
        
    x = layers.Dense(1024, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(float(l2_)))(x)    
    if normal_ == 'drop':
        x = layers.Dropout(0.25)(x)
    if normal_ == 'batch':
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_)(x)
        
    x = layers.Dense(256, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(float(l2_)))(x)
    if normal_ == 'drop':
        x = layers.Dropout(0.25)(x)
    if normal_ == 'batch':
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_)(x)
        
    x = layers.Dense(64, activation=activation_, kernel_initializer='he_normal', kernel_regularizer=l2(float(l2_)))(x)
    if normal_ == 'drop':
        x = layers.Dropout(0.25)(x)
    if normal_ == 'batch':
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation_)(x)
        
    x = layers.Dense(6, activation='softmax')(x)

    # 設定新模型的 inputs/outputs
    if image_type=='multi':
        model = Model(inputs=[base_model_1.input, base_model_2.input], outputs=x)
    else:
        model = Model(inputs=base_model.input, outputs=x)
    
    if loss_=="sparse":
        model.compile(optimizer = optimizer_f(optimizer_), 
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])    
    else:
        model.compile(optimizer = optimizer_f(optimizer_), 
                      #loss = "categorical_crossentropy", 
                      loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1),
                      metrics=["accuracy"])    
        
    return model

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--l2', type=str)
    parser.add_argument('--optimizer', type=str)  
    parser.add_argument('--pooling_type', type=str)
    parser.add_argument('--image_type', type=str)
    parser.add_argument('--loss', type=str, default='sparse')
    #parser.add_argument('--activation_conv', type=str)
    #parser.add_argument('--activation_dense', type=str)
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    l2_ = opt.l2
    optimizer_ = opt.optimizer
    pooling_ = opt.pooling_type
    image_type = opt.image_type
    loss_ = opt.loss
    
    if image_type=='merge':
        pkl_route = '06_classification_image/classification_dataset_merge.pkl'
        batch_size = 256
    if image_type=='chanel3':
        pkl_route = '06_classification_image/classification_dataset_chanel3.pkl'
        batch_size = 256
    if image_type=='multi':
        pkl_route = '06_classification_image/classification_dataset_multi.pkl'
        batch_size = 180
    if image_type=='chanel6':
        pkl_route = '06_classification_image/classification_dataset_chanel6.pkl'
        batch_size = 200
    if image_type=='chanel12':
        pkl_route = '06_classification_image/classification_dataset_chanel12.pkl'
        batch_size = 220
    if image_type=='chanel3_127':
        pkl_route = '06_classification_image/classification_dataset_chanel3_127.pkl'
        batch_size = 256
    
    try:
        with open(pkl_route, 'rb') as f:
            dataset = pickle.load(f)
    except:
        with open(pkl_route, 'rb') as f:
            dataset = joblib.load(f)

    train_X = [dataset['training_X'][x]['image'] for x in list(dataset['training_X'].keys())]
    train_X = np.array(train_X)
    test_X = [dataset['testing_X'][x]['image'] for x in list(dataset['testing_X'].keys())]
    test_X = preprocess_input(np.array(test_X))
    
    if image_type=='merge':
        train_generator = ImageDataGenerator(preprocessing_function = augmentor)
        train_generator.fit(train_X) 
    if image_type=='multi':
        train_X = [train_X[:,0,:,:,:], train_X[:,1,:,:,:]]
        test_X = [test_X[:,0,:,:,:], test_X[:,1,:,:,:]]
        for i in range(train_X[0].shape[0]):
            train_X[0][i:i+1,:,:,:] = augmentor(train_X[0][i,:,:,:])
            train_X[1][i:i+1,:,:,:] = augmentor(train_X[1][i,:,:,:])
    if 'chanel3' in image_type:
        train_X = preprocess_input(train_X)
    if image_type=='chanel6':
        for i in range(train_X[0].shape[0]):
            train_X[i:i+1,:,:,0:3] = augmentor(train_X[i,:,:,0:3])
            train_X[i:i+1,:,:,3:6] = augmentor(train_X[i,:,:,3:6])
    if image_type=='chanel12':
        for i in range(train_X.shape[0]):
            for j in range(train_X.shape[1]):
                train_X[i:i+1,j:j+1,:,:,:] = augmentor(train_X[i,j,:,:,:])
        train_X = [np.concatenate(train_X[x,:,:,:,:], axis=2) for x in range(train_X.shape[0])]
        train_X = np.array(train_X)
        test_X = [np.concatenate(test_X[x,:,:,:,:], axis=2) for x in range(test_X.shape[0])]
        test_X = np.array(test_X)
        
    if loss_=="sparse":
        train_Y = np.argmax(dataset['training_Y'], axis=1)
        test_Y = np.argmax(dataset['testing_Y'], axis=1)
        y_integers = train_Y
    else:
        train_Y = dataset['training_Y']
        test_Y = dataset['testing_Y']
        y_integers = np.argmax(train_Y, axis=1)
    del dataset
    
    class_weights = compute_class_weight('balanced', classes = np.unique(y_integers), y = y_integers)
    class_weights = dict(enumerate(class_weights))
    
    for activation_ in ["relu", "gelu", "tanh","swish"]:
        for normal_ in ["drop", "batch"]:
            if loss_=="sparse":
                files = '07_classification_training_' + image_type + "/" + optimizer_ + "/" + pooling_ + "/" + activation_ + "_" + normal_ + "_" + str(l2_)
            else:
                files = '07_classification_training_' + image_type + "_categorical" + "/" + optimizer_ + "/" + pooling_ + "/" + activation_ + "_" + normal_ + "_" + str(l2_)
            print(files)
            os.makedirs(files, exist_ok=True)
            
            if 'history.txt' not in os.listdir(files):
                model = model_build(activation_, normal_, float(l2_), optimizer_, pooling_, image_type, loss_)
                plot_model(model, to_file= files + '/Flatten.png', show_shapes=True)
                call_ = callbacks_list_fn(files, loss_, save_weights_only_ = True, earlystop = False)
                if image_type=='merge':
                    history = model.fit(train_generator.flow(train_X, train_Y, batch_size = batch_size),
                                        validation_data = (test_X, test_Y), 
                                        steps_per_epoch = len(train_X) // batch_size,
                                        class_weight = class_weights,
                                        epochs = 200, callbacks=call_)
                else:
                    history = model.fit(train_X,train_Y, 
                                        validation_data = (test_X,test_Y),
                                        class_weight = class_weights, 
                                        epochs = 200, batch_size = batch_size, callbacks=call_)  
                model.save_weights(files + '/weights_final.h5')
                
                with open(files + "/history.txt",'w') as f:
                    f.write(str(history.history))    
                # draw training curve
                with open(files + "/history.txt") as f:
                    lines = f.readlines()
                loss = list(map(lambda x:float(x), lines[0].split("], ")[0].split("[")[1].split(",")))
                accuracy = list(map(lambda x:float(x), lines[0].split("], ")[1].split("[")[1].split(",")))
                val_loss = list(map(lambda x:float(x), lines[0].split("], ")[2].split("[")[1].split(",")))
                val_accuracy = list(map(lambda x:float(x), lines[0].split("], ")[3].split("[")[1].split(",")))
                HH = [accuracy, val_accuracy, loss, val_loss]
                history_fig = show_train_history(HH)
                history_fig.savefig(files + "/history.png")
                
                tf.keras.backend.clear_session() 
                gc.collect()
