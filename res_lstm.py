import numpy as np
import tensorflow as tf
import os
#import train
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers,losses,layers,Model,Sequential
from tensorflow import keras

#from GAN import train

import tensorflow as tf

import tensorflow as tf

class Basic_Block(layers.Layer):
    def __init__(self,filter_num,kernel_size,strides=1):
        super(Basic_Block, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filter_num,kernel_size=kernel_size,strides=strides,padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation(tf.nn.leaky_relu)

        self.conv2 = layers.Conv2DTranspose(filter_num,kernel_size=3,strides=1,padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation(tf.nn.leaky_relu)

        if strides !=-1 :
            self.downsample = keras.Sequential()
            self.downsample.add(layers.Conv2DTranspose(filter_num,kernel_size=3,strides=strides,padding="same"))
            #self.downsample.add(layers.BatchNormalization(axis=-1))
        else:
            self.downsample = lambda x:x
        self.stride = strides
        #self.a = tf.Variable(0.5)  # 给生成器的每一层给与其一个变换的高斯噪声
    def call(self, inputs, **kwargs):
        residual = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        relu1 = self.relu1(conv1)
        #bn1 = self.bn1(relu1)

        conv2 = self.conv2(relu1)
        # bn2 = self.bn2(conv2)
        add = layers.add([conv2,residual])
        #add = tf.add(add,tf.random.normal(add.shape) * self.a)
        out = self.relu2(add)
        out = self.bn1(out)
        return out

class ResNet_Generator(keras.Model): #残差生成器
    def __init__(self,l=0.5):
        super(ResNet_Generator, self).__init__()
        self.layer_1 = Sequential([
            layers.Dense(2 * 2 * 1024, activation=tf.nn.leaky_relu),
            layers.BatchNormalization(),
        ])
        self.layer_2 = self.build_resblock(filter_num=int(512*l), kernel_size=1, strides=2, )
        self.layer_2_1 = layers.Conv2D(512, 3, 1, "same", activation=tf.nn.leaky_relu)
        self.layer_2_1_bn = layers.BatchNormalization()

        self.layer_3 = self.build_resblock(filter_num=int(256*l), kernel_size=5, strides=2, )
        self.layer_3_1 = layers.Conv2D(256, 5, 1, "same", activation=tf.nn.leaky_relu)
        self.layer_3_1_bn = layers.BatchNormalization()

        self.layer_4 = self.build_resblock(filter_num=int(128*l), kernel_size=5, strides=2, )
        self.layer_4_1 = layers.Conv2D(128, 5, 1, "same", activation=tf.nn.leaky_relu)
        self.layer_4_1_bn = layers.BatchNormalization()

        self.layer_5 = self.build_resblock(filter_num=int(64*l), kernel_size=5, strides=2, )
        self.layer_5_1 = layers.Conv2D(64, 5, 1, "same", activation=tf.nn.leaky_relu)
        self.layer_5_1_bn = layers.BatchNormalization()

        self.layer_6 = self.build_resblock(filter_num=3, kernel_size=5, strides=1, )
        self.layer_6_1 = layers.Conv2D(3, 5, 1, "same")
        self.layer_6_1_bn = layers.BatchNormalization()

        # self.a = tf.Variable(0.0005) # 给生成器的每一层给与其一个变换的高斯噪声
        # self.layer_8 = layers.Conv2D(3,1,1,padding="same")

    def call(self, inputs, training=None, mask=None):
        x = self.layer_1(inputs, training=training)
        x = tf.reshape(x, shape=(-1, 2, 2, 1024))

        x = self.layer_2(x, training=training)

        # x = self.layer_2_1_bn(self.layer_2_1(x),training=training)

        x = self.layer_3(x, training=training)

        # x = self.layer_3_1_bn(self.layer_3_1(x),training=training)

        x = self.layer_4(x, training=training)

        # x = self.layer_4_1_bn(self.layer_4_1(x),training=training)

        x = self.layer_5(x, training=training)

        # x = self.layer_5_1_bn(self.layer_5_1(x),training=training)

        # x = self.layer_8(x)
        x = self.layer_6(x, training=training)

        x = self.layer_6_1_bn(self.layer_6_1(x), training=training)
        x = tf.tanh(x)

        return x

    def build_resblock(self, filter_num, kernel_size, strides=1):
        res_blocks = []
        res_blocks.append(Basic_Block(filter_num, kernel_size, strides))
        # for _ in range(1, block):
        #     res_blocks.append(Basic_Block(filter_num, 1))
        return Sequential(res_blocks)

#================================================================================
class ResNet_Discriminator_Basic_Block(layers.Layer):
    def __init__(self,filter_num,kernel_size,strides=2):
        super(ResNet_Discriminator_Basic_Block, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,kernel_size=kernel_size,strides=strides,padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation(tf.nn.leaky_relu)

        self.conv2 = layers.Conv2D(filter_num,kernel_size=3,strides=1,padding="same")
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu2 = layers.Activation(tf.nn.leaky_relu)

        if strides !=-1 :
            self.downsample = keras.Sequential()
            self.downsample.add(layers.Conv2D(filter_num,kernel_size=3,strides=strides,padding="same"))
            #self.downsample.add(layers.BatchNormalization(axis=-1))
        else:
            self.downsample = lambda x:x
        self.stride = strides

    def call(self, inputs, **kwargs):
        residual = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        relu1 = self.relu1(conv1)

        #relu1 = self.relu2(self.conv2(relu1))

        add = tf.add(relu1,residual)
        #add = tf.add(add, tf.random.normal(add.shape) * self.a)
        out = self.relu2(add)
        out = self.bn1(out)
        return out

class ResNet_Discriminator(Model): # 残差卷积鉴别器
    def __init__(self,l=1):
        super(ResNet_Discriminator, self).__init__()
        self.layer_2 = self.build_resblock(filter_num=int(512*l),kernel_size=5,strides=2,)
        #self.layer_20 = self.build_resblock(filter_num=int(256 * l), kernel_size=5, strides=1, )
        self.layer_3 = self.build_resblock(filter_num=int(256*l),kernel_size=5,strides=2,)
        #self.layer_40 = self.build_resblock(filter_num=int(256*l), kernel_size=5, strides=1, )
        self.layer_41 = self.build_resblock(filter_num=int(128*l), kernel_size=5, strides=2, )
        #self.layer_42 = self.build_resblock(filter_num=64, kernel_size=5, strides=1, )
        self.layer_5 = self.build_resblock(filter_num=int(32*l), kernel_size=5, strides=2,)
        self.layer5_M = Self_Attention(int(32))
        self.layer_6_conv = keras.layers.Conv2D(int(32),1,1,"same",activation=tf.nn.leaky_relu)

        self.layer_6_fd = keras.layers.Flatten()
        #self.layer_6_bn = keras.layers.BatchNormalization()
        self.layer_7_bn = keras.layers.BatchNormalization()
        #self.lstm6 = keras.layers.LSTM(64,activation=tf.nn.leaky_relu)


        #self.layer_7_fd = keras.layers.Flatten()
        self.layer_6 = keras.layers.Dense(1,use_bias=True)

    def call(self, inputs, training=None, mask=None):
        x = self.layer_2(inputs,training=training)
        #x = self.layer_20(x, training=training)

        x = self.layer_3(x,training=training)

        #x = self.layer_4(x)
        #x = self.layer_40(x,training=training)
        x = self.layer_41(x, training=training)
        #x = self.layer_42(x, training=training)

        x_1 = self.layer_5(x,training=training)
        x_1_t = self.layer5_M(x_1,training=training)
        x_1_c = self.layer_6_conv(x_1)

        #x = self.layer_7_bn(tf.concat([x_1_c,x_1_t],axis=-1))
        x = self.layer_7_bn(tf.add(x_1_c,x_1_t))
        #x_2 = self.layer_6_bn(self.layer_6_conv(x),training=training)
        #x_2 = self.layer_6_fd(x_1)

        #b , h ,w , c = x_1.shape
        # x = tf.reshape(x_1,[b,h*w,c])

        #x = self.lstm6(x,training=training)
        #x = self.layer_7_fd(x)
        #x = tf.add(x,tf.reshape(x_1,[b,h*w,c]))

        x = self.layer_6_fd(x)

        out = self.layer_6(x,training=training)
        return out
    def build_resblock(self, filter_num, kernel_size, strides=1):
        res_blocks = []
        res_blocks.append(ResNet_Discriminator_Basic_Block(filter_num, kernel_size, strides))
        # for _ in range(1, block):
        #     res_blocks.append(Basic_Block(filter_num, 1))
        return Sequential(res_blocks)

class Self_Attention(tf.keras.Model):
    def __init__(self,  key_dim):
        super(Self_Attention, self).__init__()
        self.key_dim = key_dim

        # 定义查询、键、值的卷积层
        self.query_conv = keras.layers.Conv2D(filters=self.key_dim , kernel_size=1, strides=1, padding='same', use_bias=False)
        self.key_conv = keras.layers.Conv2D(filters=self.key_dim , kernel_size=1, strides=1, padding='same', use_bias=False)
        self.value_conv = keras.layers.Conv2D(filters=self.key_dim, kernel_size=1, strides=1, padding='same', use_bias=False)

        # 定义输出的卷积层
       # self.output_conv = keras.layers.Conv2D(self.key_dim, kernel_size=1, strides=1, padding='valid', use_bias=False)

    def call(self, inputs):
        # 将输入张量解包为查询、键、值

        # 计算查询、键、值
        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        # 计算注意力权重
        attention_scores = tf.einsum('bhwc,bwic->bhic', query, key)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, dtype=attention_scores.dtype))
        attention_weights = tf.nn.softmax(attention_scores)

        # 将值与注意力权重相乘并合并
        attention_output = attention_weights * value

        return attention_output


class Assist(Model):#设计一个引导生成器训练的方法
    def __init__(self):
        super(Assist, self).__init__()
        self.layer1 = keras.layers.Conv2D(3,1,1,"same",activation=tf.nn.relu)
        self.bn1 = keras.layers.BatchNormalization()
        self.layer2 = Self_Attention(3)
        self.bn2 = keras.layers.BatchNormalization()
        self.layer3 = keras.layers.Conv2D(3,1,1,"same",activation=tf.nn.relu)
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.bn1(self.layer1(inputs))
        x_1 = self.bn3(self.layer3(x))
        x_2 = self.bn2(self.layer2(x))
        x = tf.add(x_1,x_2)
        out = tf.tanh(x)
        return out













class Generator(Model): #普通卷积生成器
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = layers.Dense(4*4*1024,activation=tf.nn.leaky_relu)
        self.bn = layers.BatchNormalization(axis=-1)

        self.conv1 = layers.Conv2DTranspose(512, 3, 2, "same", activation=tf.nn.leaky_relu)
        self.bn1 = layers.BatchNormalization(axis=-1)

        self.conv2 = layers.Conv2DTranspose(256, 5, 2, "same", activation=tf.nn.leaky_relu)
        self.bn2 = layers.BatchNormalization(axis=-1)

        self.conv3 = layers.Conv2DTranspose(128, 5, 2, "same", activation=tf.nn.leaky_relu)
        self.bn3 = layers.BatchNormalization(axis=-1)

        self.conv4 = layers.Conv2DTranspose(3,5,2,"same",activation=tf.nn.leaky_relu)
        self.bn4 = layers.BatchNormalization(axis=-1)


    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs=inputs)
        x = self.bn(x, training=training)

        x = tf.reshape(x,shape=(-1,4,4,1024))

        x1 = self.bn1(self.conv1(x),training = training)

        x2 = self.bn2(self.conv2(x1),training = training)

        x3 = self.bn3(self.conv3(x2),training = training)

        out = self.bn4(self.conv4(x3),training = training)

        out = tf.nn.tanh(out)

        return out

class Discriminator(Model): # 普通卷积鉴别器
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(128,5,2,"valid",activation=tf.nn.relu)
        self.bn1 = layers.BatchNormalization(axis=-1)


        self.conv2 = layers.Conv2D(256,5,2,"valid",activation=tf.nn.relu)
        self.bn2 = layers.BatchNormalization(axis=-1)


        self.conv3 = layers.Conv2D(512,5,2,"valid",activation=tf.nn.relu)
        self.bn3 = layers.BatchNormalization(axis=-1)

        self.conv4 = layers.Conv2D(1024,3,2,"valid",activation=tf.nn.relu)
        self.bn4 = layers.BatchNormalization(axis=-1)

        self.fd = layers.Flatten()
        self.fc2 = layers.Dense(1)


    def call(self, inputs, training=None, mask=None):

        x1 = self.bn1(self.conv1(inputs=inputs),training = training)

        x2 = self.bn2((self.conv2(x1)),training = training)

        x3 = self.bn3((self.conv3(x2)),training = training)

        x4 = self.bn4((self.conv4(x3)), training=training)

        x4 = self.fd(x4)

        logits = self.fc2(x4)

        return logits
def generate_big_image(image_data):
    # 将前25张图片拼接成一张大图
    rows = 5
    cols = 5
    channels = 3
    image_size = 64
    big_image = np.zeros((rows * image_size, cols * image_size, channels))
    for i in range(rows):
        for j in range(cols):
            big_image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size, :] = image_data[
                i * cols + j]

    # 转换为0-255的像素值
    big_image = ((big_image + 1) / 2) * 255
    big_image = big_image.astype(np.uint8)
def predict():
    tf.random.set_seed(666)
    for i in range(100):
        generator = ResNet_Generator()
        generator.build(input_shape=(32,512))
        generator.load_weights(r"C:\Users\Arbi\PycharmProjects\tf_CV\GAN\ckpt\best_weights_g_14.h5")
        z = tf.random.normal([100, 100])
        img1 = generator(z)
        img_path = os.path.join(r"./predict", "wgan_and_sltm_normal_relu_%d.png" % i)
        img1 = generate_big_image(img1)
        #img1 = np.squeeze(img1,axis=0)
        plt.imshow(img1)
        plt.axis("off")
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()

        print(i)

if __name__ == "__main__":
    #predict()
    res = ResNet_Discriminator()
    x = tf.random.uniform([32, 64,64,3],maxval=1,minval=-1)
    y = res(x)
    res.summary()



    pass


