import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,layers
class MappingNet(tf.keras.Model):
    def __init__(self):
        super(MappingNet, self).__init__()
        self.bn = layers.BatchNormalization()
        self.mapping_net = self.build_block(block=6)
        self.out = keras.layers.Dense(512)
    def call(self, inputs, training=None, mask=None):
        assert len(inputs.shape) == 2,\
            "we shoule input size is 2d in mapping_net"
        x = self.bn(inputs)
        x = self.mapping_net(x)
        x = self.out(x)
        return x
    def build_block(self, block):
        res_blocks = []
        for _ in range(0, block):
             res_blocks.append(keras.layers.Dense(512,activation=tf.nn.relu,use_bias=True))
             #res_blocks.append(keras.layers.BatchNormalization())
        return Sequential(res_blocks)

# if __name__ == "__main__":
#     model = MappingNet()
#     model.build(input_shape=(64,512))
#     model.summary()