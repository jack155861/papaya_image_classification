import segmentation_models as sm
import os, cv2
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from random import randint

DATA_DIR = './03_unet_image'
x_train_dir = os.path.join(DATA_DIR, 'image/train')
y_train_dir = os.path.join(DATA_DIR, 'mask/train')
x_valid_dir = os.path.join(DATA_DIR, 'image/test')
y_valid_dir = os.path.join(DATA_DIR, 'mask/test')

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    CLASSES = ['papaya']
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            data_choose=False, 
    ):
        if(data_choose==False):
            self.ids = os.listdir(images_dir)
        else:
            self.ids = sum([[x+"_"+y+".jpg" for y in [str(randint(0, 31)).rjust(2,'0') for i in range(10)]] for x in list(np.unique([x.split("_")[0] for x in os.listdir(images_dir)]))], [])
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        if len(np.unique(mask))>2: # 我自己加的
            mask[mask>127] = 255
            mask[mask<=127] = 0
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

def get_preprocessing(preprocessing_fn):    
    return A.Compose([A.Lambda(image=preprocessing_fn),])

class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)       

# define network parameters
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
BACKBONE = 'resnet18'
BATCH_SIZE = 16
CLASSES = ['papaya']
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1) 
activation = 'sigmoid' if n_classes == 1 else 'softmax'
LR = 0.0001
EPOCHS = 100
preprocess_input = sm.get_preprocessing(BACKBONE)

# create model
model = sm.Unet(BACKBONE, 
                classes=n_classes, activation=activation, input_shape=(320, 640, 3), 
                encoder_weights='imagenet')
model.summary()

# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    classes=CLASSES, 
    #augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    classes=CLASSES, 
    #augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
    data_choose = True
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('04_unet_training/best_model.h5', 
                                    monitor="val_loss", mode='min', 
                                    save_best_only=True, save_weights_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min', 
                                      min_delta=0.00001, cooldown=0, threshold_mode='rel', min_lr=1e-8),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience = 25, mode='min', min_delta=0.00001)  
]

history = model.fit(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)
model.save('04_unet_training/final.h5')

# traiging fig & txt
with open('04_unet_training/history.txt','w') as f:
    f.write(str(history.history))
