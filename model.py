import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os


def build_generator(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh')(x)

    model = Model(inputs, outputs)
    return model


def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs, outputs)
    return model


def perceptual_loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg.trainable = False
    model = Model(vgg.input, vgg.get_layer('block3_conv3').output)
    return tf.reduce_mean(tf.square(model(y_true) - model(y_pred)))


def train_gan(epochs, batch_size, data_dir, checkpoint_dir, output_dir, img_shape=(128, 128, 3)):
    generator = build_generator(img_shape)
    discriminator = build_discriminator(img_shape)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False

    z = Input(shape=img_shape)
    img = generator(z)
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss=[perceptual_loss, 'binary_crossentropy'], optimizer=optimizer)

    # 체크포인트 불러오기
    checkpoint_path = os.path.join(checkpoint_dir, "gan_checkpoint")
    if os.path.exists(checkpoint_path):
        generator.load_weights(checkpoint_path)
        discriminator.load_weights(checkpoint_path)

    best_loss = float('inf')

    for epoch in range(epochs):
        for _ in range(batch_size):
            # 랜덤한 진짜 이미지를 선택
            real_images = []
            fake_images = []
            for i in range(batch_size):
                real_image = cv2.imread(os.path.join(data_dir, f"B_{i:04d}.png"))
                real_images.append(real_image)

                noise = np.random.normal(0, 1, (1, *img_shape))
                fake_image = generator.predict(noise)
                fake_images.append(fake_image)

            real_images = np.array(real_images)
            fake_images = np.array(fake_images)

            valid = np.ones((batch_size,) + (8, 8, 1))
            fake = np.zeros((batch_size,) + (8, 8, 1))

            d_loss_real = discriminator.train_on_batch(real_images, valid)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = combined.train_on_batch(noise, [real_images, valid])

        print(f"{epoch} [D loss: {d_loss[0]} | acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

        if d_loss[0] < best_loss:
            best_loss = d_loss[0]
            generator.save_weights(os.path.join(checkpoint_dir, "best_generator.h5"))
            discriminator.save_weights(os.path.join(checkpoint_dir, "best_discriminator.h5"))

        generator.save_weights(checkpoint_path)
        discriminator.save_weights(checkpoint_path)

    # 최종 영상 생성 및 저장
    result_video = []
    for i in range(100):  # 원하는 프레임 수만큼 반복
        noise = np.random.normal(0, 1, (1, *img_shape))
        generated_frame = generator.predict(noise)
        result_video.append(generated_frame[0])

    out = cv2.VideoWriter(os.path.join(output_dir, 'result_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
    for frame in result_video:
        out.write(frame.astype(np.uint8))
    out.release()


# 예시 사용법
train_gan(epochs=10000, batch_size=32, data_dir='processed_frames', checkpoint_dir='checkpoints', output_dir='output')
