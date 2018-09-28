#%%
import numpy as np
import scipy.io as sio
from PIL import Image

data = sio.loadmat(r"F:\002-4-OS.mat")

fundus = data['mosaic']

exudate = data['mosaicExudate']

#im = Image.fromarray(fundus)
im = Image.fromarray(exudate)
im.show()

##

# ------------------------------------------------------#
#%%
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)  # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()

# --------------------------------------------
#%%
import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show()

cnf_matrix = np.array([
    [4101, 2, 5, 24, 0],
    [50, 3930, 6, 14, 5],
    [29, 3, 3973, 4, 0],
    [45, 7, 1, 3878, 119],
    [31, 1, 8, 28, 3936],
])

class_names = ['Buildings', 'Farmland', 'Greenbelt', 'Wasteland', 'Water']

# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# --------------------------------------------
#%%
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# -------------------------------------------
#%%
from keras.models import load_model, Model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import cv2

model = load_model('F:\experiment\models\inception_resnet_v2_retina_1_finetune_5_increment_1_weights.h5')
# model = load_model('F:\experiment\models\inception_resnet_v2_retina_1_weights.h5')
model.summary()
#
# img_path = r"F:\experiment\data\kaggle\test-760\0\0_10149_left.jpeg"
# img_path = r"F:\experiment\data\kaggle\test-760\1\4_22669_right.jpeg"
img_path = r"F:\experiment\data\healthvision\test-370\1\7af6deb28bf34adaaf280ab5e27a4df6.jpeg"
# img_path = r"F:\experiment\data\kaggle\test-760\1\2_31867_right.jpeg"

# model = VGG16(weights='imagenet',
#               include_top=False)
# model.summary()

img = image.load_img(img_path, target_size=(512, 512))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# This is the "african elephant" entry in the prediction vector
african_elephant_output = model.output[:, 0]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
# last_conv_layer = model.get_layer('conv2d_42')
last_conv_layer = model.get_layer('conv2d_195')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(255):
# for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite(r'f:/elephant_cam.jpg', superimposed_img)

plt.show()
