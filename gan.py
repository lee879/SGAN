import numpy as np
import tensorflow as tf
import os
#import train
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers,losses,layers,Model,Sequential
from tensorflow import keras

import tensorflow as tf

import tensorflow as tf

class Transformer_Discriminator(tf.keras.Model):
    def __init__(self):
        super(Transformer_Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, 7, 2, "valid", activation=tf.nn.leaky_relu)
        self.bn1 = layers.BatchNormalization(axis=-1)

        self.conv2 = layers.Conv2D(128, 5, 2, "valid", activation=tf.nn.leaky_relu)
        self.bn2 = layers.BatchNormalization(axis=-1)

        self.conv3 = layers.Conv2D(256, 3, 2, "valid", activation=tf.nn.leaky_relu)
        self.bn3 = layers.BatchNormalization(axis=-1)

        self.conv4 = layers.Conv2D(8, 1, 1, "valid", activation=tf.nn.leaky_relu)
        self.bn4 = layers.BatchNormalization(axis=-1)

        self.rp = layers.Reshape((1,-1))
        #self.swin_transformer = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64) #在tf2.4以上版本才有这个 多头注意力
        self.latm = keras.layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2,activation=tf.tanh)

        self.fc1 = layers.Dense(16,activation=tf.nn.leaky_relu)
        self.bn5 = layers.BatchNormalization(axis=-1)
        self.fc2 = layers.Dense(1,activation=tf.nn.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        x = self.bn1(self.conv1(inputs=inputs), training=training)
        x = self.bn2((self.conv2(x)), training=training)
        x = self.bn3((self.conv3(x)), training=training)
        x = self.bn4((self.conv4(x)), training=training)
        #x = self.rp(x)

        #x = self.latm(x)
        x = self.bn5(self.fc1(x), training=training)
        x = self.fc2(x)

        return x

class Basic_Block(layers.Layer):
    def __init__(self,filter_num,kernel_size,strides=1):
        super(Basic_Block, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filter_num,kernel_size=kernel_size,strides=strides,padding="same")
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.relu1 = layers.Activation(tf.nn.leaky_relu)

        self.conv2 = layers.Conv2DTranspose(filter_num,kernel_size=1,strides=1,padding="same")
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu2 = layers.Activation(tf.nn.leaky_relu)

        if strides !=-1 :
            self.downsample = keras.Sequential()
            self.downsample.add(layers.Conv2DTranspose(filter_num,kernel_size=kernel_size,strides=strides,padding="same",activation=tf.nn.leaky_relu))
            self.downsample.add(layers.BatchNormalization(axis=-1))
        else:
            self.downsample = lambda x:x
        self.stride = strides

    def call(self, inputs, **kwargs):
        residual = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        relu1 = self.relu1(conv1)
        bn1 = self.bn1(relu1)

        conv2 = self.conv2(bn1)
        bn2 = self.bn2(conv2)
        add = layers.add([bn2,residual])
        out = self.relu2(add)
        return out

class ResNet_Generator(keras.Model): #残差生成器
    def __init__(self,classes_num=100):
        super(ResNet_Generator, self).__init__()
        self.layer_1 = Sequential([
            layers.Dense(4*4*256,activation=tf.nn.leaky_relu),
            layers.BatchNormalization(axis=-1),
        ])
        self.layer_2 = self.build_resblock(filter_num=128,kernel_size=1,strides=4,)
        self.layer_3 = self.build_resblock(filter_num=32,kernel_size=3,strides=2,)
        self.layer_4 = self.build_resblock(filter_num=16,kernel_size=5,strides=2,)
        self.layer_5 = self.build_resblock(filter_num=8, kernel_size=7, strides=1,)
        self.layer_8 = layers.Conv2D(3,1,1,padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.layer_1(inputs,training=training)
        x = tf.reshape(x, shape=(-1, 4, 4, 256))
        x = self.layer_2(x,training=training)
        x = self.layer_3(x,training=training)
        x = self.layer_4(x,training=training)
        x = self.layer_5(x,training=training)
        x = self.layer_8(x,training=training)

        x = tf.tanh(x)
        return x

    def build_resblock(self, filter_num,kernel_size ,strides=1):
        res_blocks = []
        res_blocks.append(Basic_Block(filter_num,kernel_size ,strides))
        # for _ in range(1, block):
        #     res_blocks.append(Basic_Block(filter_num, 1))
        return Sequential(res_blocks)









class Generator(Model): #普通卷积生成器
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = layers.Dense(3*3*512,activation=tf.nn.leaky_relu)
        self.dp = layers.Dropout(0.2)
        self.bn = layers.BatchNormalization(axis=-1)


        self.conv1 = layers.Conv2DTranspose(256, 3, 3, "valid", activation=tf.nn.leaky_relu)
        self.dp1 = layers.Dropout(0.2)
        self.bn1 = layers.BatchNormalization(axis=-1)

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, "valid", activation=tf.nn.leaky_relu)
        self.dp2 = layers.Dropout(0.2)
        self.bn2 = layers.BatchNormalization(axis=-1)

        self.conv3 = layers.Conv2DTranspose(64, 4, 3, "valid", activation=tf.nn.leaky_relu)
        self.dp3 = layers.Dropout(0.2)
        self.bn3 = layers.BatchNormalization(axis=-1)

        self.conv4_0 = layers.Conv2D(32,1,1,"same",activation="relu")
        self.conv4 = layers.Conv2D(3,1,1,"same")

    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs=inputs)
        x = self.bn(x, training=training)
        x = self.dp(x)

        x = tf.reshape(x,shape=(-1,3,3,512))

        x1 = self.dp1(self.bn1(self.conv1(x)),training = training)

        x2 = self.dp2(self.bn2(self.conv2(x1)),training = training)

        x3 = self.dp3(self.bn3(self.conv3(x2)),training = training)

        x3 = tf.reshape(x3,shape=(-1,64,64,3))

        x3 = self.conv4_0(x3,training = training)
        out = self.conv4(x3,training = training)

        out = tf.nn.tanh(out)

        return out

class Discriminator(Model): # 普通卷积鉴别器
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64,3,3,"valid",activation=tf.nn.relu)
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.dp1 = layers.Dropout(0.1)

        self.conv2 = layers.Conv2D(128,3,2,"valid",activation=tf.nn.relu)
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.dp2 = layers.Dropout(0.1)

        self.conv3 = layers.Conv2D(256,3,3,"valid",activation=tf.nn.relu)
        self.bn3 = layers.BatchNormalization(axis=-1)

        self.conv4 = layers.Conv2D(256,1,1,"valid",activation=tf.nn.relu)
        self.bn4 = layers.BatchNormalization(axis=-1)


        self.fd = layers.Flatten()
        self.fc2 = layers.Dense(1)


    def call(self, inputs, training=None, mask=None):

        x1 = self.bn1(self.dp1(self.conv1(inputs=inputs),training = training))

        x2 = self.bn2(self.dp2((self.conv2(x1)),training = training))

        x3 = self.bn3((self.conv3(x2)),training = training)

        x4 = self.bn4((self.conv4(x3)), training=training)


        x4 = self.fd(x4)



        logits = self.fc2(x4)

        return logits

def predict():
    tf.random.set_seed(666)
    for i in range(100):
        generator = ResNet_Generator()
        generator.build(input_shape=(None,100))
        generator.load_weights(r"C:\Users\Arbi\PycharmProjects\tf_CV\GAN\ckpt\best_weights.h5")
        z = tf.random.uniform([100, 100])
        img1 = generator(z)
        img_path = os.path.join(r"./predict", "wgan_and_sltm_normal_relu_%d.png" % i)
        img1 = np.squeeze(img1,axis=0)
        plt.imshow(img1)
        plt.axis("off")
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()

        print(i)

if __name__ == "__main__":
    #predict()
    res = ResNet_Generator()
    x = tf.random.uniform([100, 64],maxval=1,minval=-1)
    y = res(x)
    print("-------------------------")
    print(y.shape)


    pass


