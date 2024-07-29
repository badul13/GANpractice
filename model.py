import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

def build_generator(input_shape):
    inputs = Input(shape=input_shape)

    # Downsampling
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

    # Upsampling
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

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

    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

def perceptual_loss(y_true, y_pred):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    vgg.trainable = False
    model = Model(vgg.input, vgg.get_layer('block3_conv3').output)
    return tf.reduce_mean(tf.square(model(y_true) - model(y_pred)))

def load_and_preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1
    return img

def load_batch(data_dir, batch_size):
    files = [f for f in os.listdir(data_dir) if f.startswith('B_')]
    np.random.shuffle(files)
    batch_files = files[:batch_size]
    batch_images = [load_and_preprocess_image(os.path.join(data_dir, f)) for f in batch_files]
    return np.array(batch_images)

def train_gan(epochs, batch_size, data_dir, checkpoint_dir, output_dir):
    img_shape = (1080, 1920, 3)
    print(f"Image shape: {img_shape}")

    generator = build_generator(img_shape)
    discriminator = build_discriminator(img_shape)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = Adam(0.0002, 0.5)

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False

    z = Input(shape=img_shape)
    img = generator(z)
    valid = discriminator(img)

    combined = Model(z, [img, valid])
    combined.compile(loss=[perceptual_loss, 'binary_crossentropy'], loss_weights=[1.0, 1.0], optimizer=optimizer)

    # 체크포인트 불러오기
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
                'D acc': 100 * d_loss[1],
                'G loss': g_loss[0],
                'Perceptual loss': g_loss[1]
            })

        if d_loss[0] < best_loss:
            best_loss = d_loss[0]
            generator.save_weights(os.path.join(checkpoint_dir, "best_generator.h5"))
            discriminator.save_weights(os.path.join(checkpoint_dir, "best_discriminator.h5"))

        generator.save_weights(checkpoint_path)
        discriminator.save_weights(checkpoint_path)

    # 최종 영상 생성 및 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_video = []
    for i in range(100):  # 원하는 프레임 수만큼 반복
        noise = np.random.normal(0, 1, (1,) + img_shape)
        generated_frame = generator.predict(noise)
        result_video.append(generated_frame[0])

    out = cv2.VideoWriter(os.path.join(output_dir, 'result_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_shape[1], img_shape[0]))
    for frame in result_video:
        out.write(((frame + 1) * 127.5).astype(np.uint8))
    out.release()

# 배치 크기를 조정
train_gan(epochs=100, batch_size=8, data_dir='processed_frames', checkpoint_dir='checkpoints', output_dir='output')