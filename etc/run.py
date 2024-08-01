import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.core.display_functions import clear_output
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_generator():
    model = tf.keras.Sequential([
        Input(shape=(100,)),
        tf.keras.layers.Dense(256 * 16 * 16),
        tf.keras.layers.Reshape((16, 16, 256)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True
)
generator_optimizer = Adam(learning_rate=lr_schedule)
discriminator_optimizer = Adam(learning_rate=lr_schedule)

BATCH_SIZE = 64
EPOCHS = 10000
BUFFER_SIZE = 60000

# 데이터 로드
augmented_data_frames = np.load('preprocessed_data.npy')
logging.info(f"Loaded preprocessed data from 'preprocessed_data.npy'.")

dataset = tf.data.Dataset.from_tensor_slices(augmented_data_frames).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

def gradient_penalty(batch_size, real_images, fake_images):
    epsilon = tf.random.uniform([tf.shape(real_images)[0], 1, 1, 1], 0.0, 1.0)
    real_images = tf.cast(real_images, dtype=tf.float32)
    fake_images = tf.cast(fake_images, dtype=tf.float32)

    # 배치 크기 맞추기
    min_batch_size = tf.minimum(tf.shape(real_images)[0], tf.shape(fake_images)[0])
    real_images = real_images[:min_batch_size]
    fake_images = fake_images[:min_batch_size]

    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_images_pred = discriminator(interpolated_images, training=True)

    gradients = tape.gradient(interpolated_images_pred, interpolated_images)
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
80





@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        gp = gradient_penalty(BATCH_SIZE, images, generated_images)
        total_disc_loss = disc_loss + 10.0 * gp

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def save_model(generator, discriminator, model_dir='./models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gen_weights_path = os.path.join(model_dir, 'generator.weights.h5')
    disc_weights_path = os.path.join(model_dir, 'discriminator.weights.h5')

    generator.save_weights(gen_weights_path)
    discriminator.save_weights(disc_weights_path)

def load_models(model_dir='./models'):
    gen_weights_path = os.path.join(model_dir, 'generator.weights.h5')
    disc_weights_path = os.path.join(model_dir, 'discriminator.weights.h5')

    generator = build_generator()
    discriminator = build_discriminator()

    if os.path.exists(gen_weights_path) and os.path.exists(disc_weights_path):
        generator.load_weights(gen_weights_path)
        discriminator.load_weights(disc_weights_path)
        return generator, discriminator
    else:
        return None, None

def train(dataset, epochs, model_dir='./models', save_every=10, log_every=50):
    generator, discriminator = load_models(model_dir)
    if generator is None or discriminator is None:
        generator = build_generator()
        discriminator = build_discriminator()

    seed = tf.random.normal([16, 100])
    start_time = time.time()
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        if (epoch + 1) % log_every == 0:
            logging.info(f'Epoch {epoch + 1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}')

        if (epoch + 1) % save_every == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            save_model(generator, discriminator, model_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f'Training finished in {elapsed_time:.2f} seconds.')

train(dataset, EPOCHS)
