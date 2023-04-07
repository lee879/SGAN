
import os
import numpy as np
import tensorflow as tf
from Synthesis_net import S_Net

from tensorflow.keras.callbacks import ModelCheckpoint ,Callback
#from sklearn.externals._pilutil import toimage
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import math
from _pilutil import toimage
import matplotlib.pyplot as plt
import glob
#from gan import Discriminator, ResNet_Generator, Transformer_Discriminator
from dataset import make_anime_dataset
#from dcgan import Generator,Discriminator,LSTM_Discriminator
from res_lstm import ResNet_Discriminator,ResNet_Generator

# class CosineAnnealingScheduler(Callback):
#     def __init__(self, T_max, eta_max, eta_min=0, verbose=1):
#         self.T_max = T_max
#         self.eta_max = eta_max
#         self.eta_min = eta_min
#         self.verbose = verbose
#         self.current_epoch = 0
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.current_epoch = epoch
#
#     def on_batch_begin(self, batch, model=None,logs=None):
#         if not hasattr(model.optimizer, 'lr'):
#             raise ValueError('Optimizer must have a "lr" attribute.')
#         lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
#         tf.keras.backend.set_value(self.model.optimizer.lr, lr)
#         if self.verbose > 0:
#             print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
#                   'rate to %s.' % (self.current_epoch + 1, lr))


def generate_big_image(image_data):
    # 将前25张图片拼接成一张大图
    rows =3
    cols = 3
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

    return np.expand_dims(big_image,axis=0)
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)

def celoss_ones(logits):
    #热编码
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def  d_loss_fn(generator,discriminator,batch_z,batch_xy,is_training):
    # treat real image  as real
    # treat generator image as fake
    fake_image = generator(batch_z,True)
    d_fake_logits = discriminator(fake_image,is_training)
    d_real_logits = discriminator(batch_xy,is_training)
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator,batch_xy,fake_image)
    loss = d_loss_real + d_loss_fake + 5 * gp

    return loss,gp
def g_loss_fn(generator,discriminator,batch_z,batch_xy,is_training):
    fake_img = generator(batch_z,is_training)
    d_fake_logits = discriminator(fake_img,True)
    #gp = gradient_penalty(discriminator, batch_xy, fake_img)

    loss = celoss_ones(d_fake_logits)
    return loss , fake_img

def gradient_penalty(discriminator,batch_xy,fake_image): # wgan主要的贡献
    t = tf.random.uniform(batch_xy.shape,minval=0,maxval=1)
    #t = tf.random.normal(batch_xy.shape, mean=0., stddev=1.)
    #t = tf.random.uniforml(batch_xy.shape,minval=-1,maxval=1)
    interplate = t * batch_xy + (1 - t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits,interplate)
    #grads[b,h,w,c]
    grads = tf.reshape(grads,[grads.shape[0],-1])#来进行一个打平的操作
    gp = tf.norm(grads,axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp

#使用余弦退火降低学习率
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T):
        super(CosineAnnealingSchedule, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T

    def __call__(self, step):

        t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((step/self.T) * np.pi))
        print("step{},lr;{}".format(step,t))
        return t

def main():
    # tf.random.set_seed(666)
    # np.random.seed(666)
    best_g_loss = 9999999999999.
    # hyper parameters
    z_dim = 512
    epochs = 888888
    batch_size = 24 # 更具显存换批次跑()
    is_training = True
    summary_writer = tf.summary.create_file_writer(r"D:\log")

    # data
    img_path = glob.glob(r".\datas\anime_faces\*.png")
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)  # 自己建立的数据划分
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    #generator = ResNet_Generator()  # 使用的残差生成器
    generator = S_Net(batch=batch_size)
    generator.build(input_shape=(batch_size, z_dim))
    #a = generator.load_weights(r"C:\Users\Arbi\PycharmProjects\tf_CV\GAN\ckpt\best_weights.h5")
    #generator.save_weights(a)
    #discriminator = Discriminator()
    #discriminator = LSTM_Discriminator()
    discriminator = ResNet_Discriminator()

    checkpoint_dir = './ckpt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Define the checkpoint to save the model with the best weights based on validation loss
    #best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_weights_d_14.h5')

    best_weights_checkpoint_path_g = os.path.join(checkpoint_dir, 'best_weights_g_18.h5')
    best_weights_checkpoint_g = ModelCheckpoint(best_weights_checkpoint_path_g,
                                              monitor='loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min')

    # best_weights_checkpoint_d = ModelCheckpoint(best_weights_checkpoint_path_d,
    #                                             monitor='loss',
    #                                             verbose=1,
    #                                             save_best_only=True,
    #                                             save_weights_only=True,
    #                                             mode='min')

    # lr_schedule = CosineAnnealingSchedule(lr_max=0.001, lr_min=0.00001, T=epochs)
    # lr_schedule_g = CosineAnnealingSchedule(lr_max=0.001, lr_min=0.00001, T=epochs)

    temp_d_loss = 0
    for epoch in range(epochs):

        batch_z = tf.random.normal([batch_size, z_dim], mean=0., stddev=1.)
        batch_xy = next(db_iter)


        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_xy, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        tf.optimizers.Adam(learning_rate=5e-4, beta_1=0.5).apply_gradients(zip(grads, discriminator.trainable_variables))

        # if tf.math.abs(d_loss - temp_d_loss) <= 10: #loss波动约束
        with tf.GradientTape() as tape:
            g_loss, fake_img = g_loss_fn(generator, discriminator, batch_z, batch_xy, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        tf.optimizers.Adam(learning_rate=5.1e-4, beta_1=0.9).apply_gradients(zip(grads, generator.trainable_variables))
        temp_d_loss = float(d_loss)
        print(epoch, "d_loss:", temp_d_loss, "g_loss", float(g_loss))

        with summary_writer.as_default():
            tf.summary.scalar('d_loss', float(d_loss), step=epoch)
            tf.summary.scalar('g_loss', float(g_loss), step=epoch)
            img1 = generate_big_image(fake_img)
            tf.summary.image("fake_image", img1, step=epoch)
            img2 = generate_big_image(batch_xy)
            tf.summary.image("real_image", img2, step=epoch)
            tf.summary.trace_on(graph=True, profiler=False)

            tf.summary.trace_export(name="generator_trace",step=epoch)
            summary_writer.flush()


        if epoch % 100 == 0:
            print(epoch, "d_loss:", float(d_loss), "g_loss", float(g_loss), "all_loss:", float(g_loss + d_loss),
                  "gp:",
                  float(gp))
            generator.save_weights(best_weights_checkpoint_path_g)
            #discriminator.save_weights(best_weights_checkpoint_path_d)
            # z = tf.random.normal([100,100],mean=0,stddev=1)
            # fake_iamge = generator(z,training=False)
            #img_path = os.path.join(r"./photo", "wgan_and_sltm_normal_relu_%d.png" % epoch)
            # img1 = np.squeeze(img1,axis=0)
            # plt.imshow(img1)
            # plt.axis("off")
            # plt.savefig(img_path, bbox_inches="tight")
            # plt.close()

def predict():
    generator = Generator()
    generator.build(input_shape=(100, 100))
    generator.load_weights(r"C:\Users\Arbi\PycharmProjects\tf_CV\GAN\ckpt\best_weights.h5")
    z = tf.random.normal([100,100],mean=0,stddev=1)
    fake_iamge = generator(z, training=False)
    img_path = os.path.join(r"./predict", "testv4.png" )
    save_result(fake_iamge.numpy(), 5, img_path, color_mode="P")

if __name__ == "__main__":
    main()
    #predict()
    # generator.load_weights(best_model_checkpoint_path)