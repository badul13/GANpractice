from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Activation, Concatenate, Dropout, Cropping2D, Flatten, Dense
from keras.models import Model
import numpy as np
import cv2
import os
from tqdm import tqdm
from keras.applications import VGG19
import tensorflow as tf

# VGG19 모델을 전역 변수로 한 번만 생성
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(1088, 1920, 3))
vgg.trainable = False
vgg_output = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)

# Perceptual loss 함수 정의
def perceptual_loss(y_true, y_pred):
    y_true_features = vgg_output(y_true)
    y_pred_features = vgg_output(y_pred)
    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

def build_generator(img_shape):
    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same', activation='relu')(layer_input)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        h_diff = u.shape[1] - skip_input.shape[1]
        w_diff = u.shape[2] - skip_input.shape[2]
        u = Cropping2D(((h_diff // 2, h_diff - h_diff // 2), (w_diff // 2, w_diff - w_diff // 2)))(u)
        u = Concatenate()([u, skip_input])
        return u

    d0 = Input(shape=img_shape)

    d1 = conv2d(d0, 64, bn=False)
    d2 = conv2d(d1, 128)
    d3 = conv2d(d2, 256)
    d4 = conv2d(d3, 512)
    d5 = conv2d(d4, 512)
    d6 = conv2d(d5, 512)
    d7 = conv2d(d6, 512)

    u1 = deconv2d(d7, d6, 512)
    u2 = deconv2d(u1, d5, 512)
    u3 = deconv2d(u2, d4, 512)
    u4 = deconv2d(u3, d3, 256)
    u5 = deconv2d(u4, d2, 128)
    u6 = deconv2d(u5, d1, 64)

    u7 = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(u6)
    output_img = Conv2D(img_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

def build_discriminator(img_shape):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img = Input(shape=img_shape)

    d1 = d_layer(img, 64, bn=False)
    d2 = d_layer(d1, 128)
    d3 = d_layer(d2, 256)
    d4 = d_layer(d3, 512)
    d5 = d_layer(d4, 512)

    validity = Flatten()(d5)
    validity = Dense(1, activation='sigmoid')(validity)

    return Model(img, validity)

def load_batch(data_dir, batch_size):
    images = []
    for img_name in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 127.5 - 1.0  # Normalize to [-1, 1]
        images.append(img)
    images = np.array(images)
    idx = np.random.randint(0, images.shape[0], batch_size)
    return images[idx]

def train_gan(epochs, batch_size, data_dir, checkpoint_dir, output_dir):
    img_shape = (1088, 1920, 3)
    print(f"Image shape: {img_shape}")

    generator = build_generator(img_shape)
    discriminator = build_discriminator(img_shape)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    optimizer.build(generator.trainable_variables + discriminator.trainable_variables)

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False

    z = Input(shape=img_shape)
    img = generator(z)
    valid = discriminator(img)

    combined = Model(z, [img, valid])
    combined.compile(loss=[perceptual_loss, 'binary_crossentropy'], loss_weights=[1.0, 1.0], optimizer=optimizer)

    checkpoint_path = os.path.join(checkpoint_dir, "gan_checkpoint")
    if os.path.exists(checkpoint_path + '.index'):
        generator.load_weights(checkpoint_path)
        discriminator.load_weights(checkpoint_path)

    best_loss = float('inf')

    for epoch in range(epochs):
        progress_bar = tqdm(range(batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for _ in progress_bar:
            real_images = load_batch(data_dir, batch_size)
            noise = np.random.normal(0, 1, (batch_size,) + img_shape)
            fake_images = generator.predict(noise)

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_images, valid)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = combined.train_on_batch(noise, [real_images, valid])

            progress_bar.set_postfix({
                'D loss': d_loss[0],
                'D acc': d_loss[1],
                'G loss': g_loss[0],
                'Perceptual loss': g_loss[1]
            })

        if d_loss[0] < best_loss:
            best_loss = d_loss[0]
            generator.save_weights(os.path.join(checkpoint_dir, "best_generator.h5"))
            discriminator.save_weights(os.path.join(checkpoint_dir, "best_discriminator.h5"))

        generator.save_weights(checkpoint_path)
        discriminator.save_weights(checkpoint_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_video = []
    for i in range(100):
        noise = np.random.normal(0, 1, (1,) + img_shape)
        generated_frame = generator.predict(noise)
        result_video.append(generated_frame[0])

    out = cv2.VideoWriter(os.path.join(output_dir, 'result_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_shape[1], img_shape[0]))
    for frame in result_video:
        out.write(((frame + 1) * 127.5).astype(np.uint8))
    out.release()

train_gan(epochs=100, batch_size=4, data_dir='processed_frames', checkpoint_dir='checkpoints', output_dir='output')
