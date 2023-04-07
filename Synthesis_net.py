import tensorflow as tf
from tensorflow import keras
from AN import Style,AdaIN
from Mapping_Net import MappingNet

class S_Net_Const(keras.layers.Layer):
    def __init__(self,batch):
        super(S_Net_Const, self).__init__()
        self.const = tf.Variable(tf.random.normal(shape=(batch,4,4,512),mean=0,stddev=1),trainable=False)

    def call(self, inputs, training=None, mask=None):
        return self.const + self.nosi(self.const)

    def nosi(self,inputs):
        return tf.random.normal(inputs.shape,mean=0,stddev=1)

class nosi(keras.layers.Layer):
    def __init__(self):
        super(nosi, self).__init__()
        self.w = self.add_variable("w",trainable=True)

    def call(self, inputs, **kwargs):
        noise = tf.random.normal(inputs.shape, mean=0, stddev=1) * self.w
        return noise + inputs

class S_Net(keras.Model):
    def __init__(self,batch,ini_batch=512):
        super(S_Net, self).__init__()
        self.mapNet = MappingNet()

        self.const = S_Net_Const(batch=batch)
        self.style1 = Style(output=int(ini_batch))
        self.an1 = AdaIN()
        self.conv1 = keras.layers.Conv2D(512,3,1,"same",activation=tf.nn.relu)
        self.style2 = Style(output=int(ini_batch))
        self.an2 = AdaIN()
        self.n1 = nosi()

        self.upsample = keras.layers.UpSampling2D(2)
        self.conv2 = keras.layers.Conv2D(256, 5, 1, "same", activation=tf.nn.relu)
        self.style3 = Style(output=int(ini_batch / 2))
        self.an3 = AdaIN()
        self.conv3 = keras.layers.Conv2D(256, 5, 1, "same", activation=tf.nn.relu)
        self.style4 = Style(output=int(ini_batch / 2))
        self.an4 = AdaIN()
        self.n2 = nosi()
        self.n22 = nosi()

        self.upsample1 = keras.layers.UpSampling2D(2)
        self.conv4 = keras.layers.Conv2D(128, 5, 1, "same", activation=tf.nn.relu)
        self.style5 = Style(output=int(ini_batch / 4))
        self.an5 = AdaIN()
        self.conv5 = keras.layers.Conv2D(128, 5, 1, "same", activation=tf.nn.relu)
        self.style6 = Style(output=int(ini_batch / 4))
        self.an6 = AdaIN()
        self.n3 = nosi()
        self.n33 = nosi()

        self.upsample2 = keras.layers.UpSampling2D(2)
        self.conv6 = keras.layers.Conv2D(64, 5, 1, "same", activation=tf.nn.relu)
        self.style7 = Style(output=int(ini_batch / 8))
        self.an7 = AdaIN()
        self.conv7 = keras.layers.Conv2D(64, 5, 1, "same", activation=tf.nn.relu)
        self.style8 = Style(output=int(ini_batch / 8))
        self.an8 = AdaIN()
        self.n4 = nosi()
        self.n44 = nosi()

        self.upsample3 = keras.layers.UpSampling2D(2)
        self.conv8 = keras.layers.Conv2D(32, 5, 1, "same", activation=tf.nn.relu)
        self.style9 = Style(output=int(ini_batch / 16))
        self.an9 = AdaIN()
        self.conv9 = keras.layers.Conv2D(32, 5, 1, "same", activation=tf.nn.relu)
        self.style10 = Style(output=int(ini_batch / 16))
        self.an10 = AdaIN()
        self.n5 = nosi()
        self.n55 = nosi()
        self.out = keras.layers.Conv2D(3, 1, 1, "same")

    def call(self, inputs ,training=None, mask=None):
        x = self.const(inputs=None) # 一个基板
        w = self.mapNet(inputs)
        alpha , beta = self.style1(w) # 输入的是一个噪声得到的是一个style
        out1 = self.an1(inputs=x,alpha=alpha,beta=beta)
        out1 = self.conv1(out1)
        out1 = self.n1(out1)
        alpha , beta = self.style2(w) # 输入的是一个噪声得到的是一个style
        out1 = self.an2(inputs=out1,alpha=alpha,beta=beta)
        # out1 输出的是[64,4,4,512]

        out2 = self.upsample(out1)
        out2 = self.conv2(out2)
        out2 = self.n2(out2)
        alpha, beta = self.style3(w)  # 输入的是一个噪声得到的是一个style
        out2 = self.an3(inputs=out2, alpha=alpha, beta=beta)
        out2 = self.conv3(out2)
        out2 = self.n22(out2)
        alpha, beta = self.style4(w)  # 输入的是一个噪声得到的是一个style
        out2 = self.an4(inputs=out2, alpha=alpha, beta=beta)
        # out2 输出的是[64,8,8,256]

        out3 = self.upsample1(out2)
        out3 = self.conv4(out3)
        out3 = self.n3(out3)
        alpha, beta = self.style5(w)  # 输入的是一个噪声得到的是一个style
        out3 = self.an5(inputs=out3, alpha=alpha, beta=beta)
        out3 = self.conv5(out3)
        out3 = self.n33(out3)
        alpha, beta = self.style6(w)  # 输入的是一个噪声得到的是一个style
        out3 = self.an6(inputs=out3, alpha=alpha, beta=beta)
        # out3 输出的是[64,16,16,128]

        out4 = self.upsample2(out3)
        out4 = self.conv6(out4)
        out4 = self.n4(out4)
        alpha, beta = self.style7(w)  # 输入的是一个噪声得到的是一个style
        out4 = self.an7(inputs=out4, alpha=alpha, beta=beta)
        out4 = self.conv7(out4)
        out4 = self.n44(out4)
        alpha, beta = self.style8(w)  # 输入的是一个噪声得到的是一个style
        out4 = self.an8(inputs=out4, alpha=alpha, beta=beta)
        # out4 输出的是[64,32,32,64]

        out5 = self.upsample3(out4)
        out5 = self.conv8(out5)
        out5 = self.n5(out5)
        alpha, beta = self.style9(w)  # 输入的是一个噪声得到的是一个style
        out5 = self.an9(inputs=out5, alpha=alpha, beta=beta)
        out5 = self.conv9(out5)
        out5 = self.n55(out5)
        alpha, beta = self.style10(w)  # 输入的是一个噪声得到的是一个style
        out5 = self.an10(inputs=out5, alpha=alpha, beta=beta)
        # out5 输出的是[64,64,64,32]

        out = tf.tanh(self.out(out5))
        #out 输出的是[64,64,64,3]
        #print(out.shape)

        return out

    # def nosi(self,inputs):
    #     #nosi = tf.Variable(tf.random.normal(inputs.shape, mean=0, stddev=1), trainable=True)
    #     noise = tf.random.normal(inputs.shape, mean=0, stddev=1)
    #     return noise

if __name__ == "__main__":
    model =  S_Net(batch=32)
    x = tf.random.normal(shape=(32,512))
    y = model(x)
    model.summary()



