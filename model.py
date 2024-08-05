import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Concatenate, Dropout, \
    Cropping2D, Flatten, Dense
from keras.models import Model
from keras.applications import VGG19
from tqdm import tqdm
from mtcnn import MTCNN  # mtcnn 임포트

# VGG19 모델 생성
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

    d1 = conv2d(d0, 32, bn=False)
    d2 = conv2d(d1, 64)
    d3 = conv2d(d2, 128)
    d4 = conv2d(d3, 256)
    d5 = conv2d(d4, 256)
    d6 = conv2d(d5, 256)

    u1 = deconv2d(d6, d5, 256)
    u2 = deconv2d(u1, d4, 256)
    u3 = deconv2d(u2, d3, 128)
    u4 = deconv2d(u3, d2, 64)
    u5 = deconv2d(u4, d1, 32)

    u6 = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')(u5)
    output_img = Conv2D(img_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(u6)

    return Model(d0, output_img)

def build_discriminator(img_shape):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img = Input(shape=img_shape)

    d1 = d_layer(img, 32, bn=False)
    d2 = d_layer(d1, 64)
    d3 = d_layer(d2, 128)
    d4 = d_layer(d3, 256)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='valid')(d4)
    validity = Flatten()(validity)
    validity = Dense(1, activation='sigmoid')(validity)

    return Model(img, validity)

def load_batch(data_dir, batch_size):
    detector = MTCNN()  # mtcnn 인스턴스 생성
    image_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
                   fname.endswith('.jpg') or fname.endswith('.png')]
    np.random.shuffle(image_files)

    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        if len(faces) == 0:
            print(f"Warning: No faces detected in image {img_path}")
            continue
        for face in faces:
            x, y, width, height = face['box']
            face_img = img[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, (1920, 1088))
            face_img = face_img / 127.5 - 1.0  # Normalize to [-1, 1]
            images.append(face_img)
        if len(images) >= batch_size:
            break

    if len(images) < batch_size:
        raise ValueError(
            "Not enough images to create a batch. Please check the data directory and face detection logic.")

    return np.array(images[:batch_size])

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
        progress_bar = tqdm(range(batch_size), desc=f"Epoch {epoch + 1}/{epochs}")
        for _ in progress_bar:
            try:
                real_images = load_batch(data_dir, batch_size)
            except ValueError as e:
                print(e)
                continue

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_video = []
    for i in range(100):
        noise = np.random.normal(0, 1, (1,) + img_shape)
        generated_frame = generator.predict(noise)
        result_video.append(generated_frame[0])

    out = cv2.VideoWriter(os.path.join(output_dir, 'result_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (img_shape[1], img_shape[0]))
    for frame in result_video:
        out.write(((frame + 1) * 127.5).astype(np.uint8))
    out.release()

if __name__ == "__main__":
    train_gan(epochs=10, batch_size=8, data_dir='processed_frames', checkpoint_dir='checkpoints', output_dir='output')
