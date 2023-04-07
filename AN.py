import tensorflow as tf
from tensorflow import keras
class style(tf.keras.layers.Layer):
    def __init__(self, batch,epsilon=1e-5):
        super(Style, self).__init__()
        self.layer_s = keras.layers.Dense(batch,use_bias=True,activation=tf.nn.relu)
        self.layer_s_bn = keras.layers.BatchNormalization()
        self.layer_b = keras.layers.Dense(batch,use_bias=True,activation=tf.nn.relu)
        self.layer_b_bn = keras.layers.BatchNormalization()
    def call(self, inputs):
        # Separate content and style features
        ys = self.layer_s_bn(self.layer_s(inputs))

        yb = self.layer_b_bn(self.layer_b(inputs))
        alpha = tf.expand_dims(tf.expand_dims(ys, axis=1), axis=1)
        beta = tf.expand_dims(tf.expand_dims(yb, axis=1), axis=1)
        return alpha , beta
class Style(tf.keras.layers.Layer):
    def __init__(self,output):
        super(Style, self).__init__()
        self.up = tf.keras.layers.UpSampling1D(2)
        self.conv1d = tf.keras.layers.Conv1D(128,1,1,padding="same",activation=tf.nn.leaky_relu)
        self.conv2d = tf.keras.layers.Conv2D(128,1,1,padding = "same",activation=tf.nn.leaky_relu)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc =  tf.keras.layers.Dense(output)

    def call(self, inputs, training=False, mask=None):
        x = tf.expand_dims(inputs,axis=1)
        x = self.up(x)
        x = self.conv1d(x)
        x = tf.expand_dims(x, axis=2)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.fc(x)
        return tf.expand_dims(x[:,0,:,:],axis=1),tf.expand_dims(x[:,1,:,:],axis=1)

class AdaIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
    def call(self, inputs,alpha,beta, **kwargs):
        mean_content, variance_content = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        out = alpha * ((inputs - mean_content) / tf.math.sqrt(variance_content ** 2 + self.epsilon )) + beta
        return out

# if __name__ == "__main__":
#     # Define content and style features
#
#     test = tf.random.normal([64,512])
#     test2 = tf.random.uniform([64,4,4,512])
#     # Apply AdaIN normalization
#     Style = Style()
#     alpha,beta = Style(test)
#     aadain = AdaIN()
#     adain = aadain(inputs=test2,alpha=alpha,beta=beta)
#     print(adain.shape)








