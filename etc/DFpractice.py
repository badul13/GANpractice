import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.core.display_functions import clear_output
from imutils import face_utils
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dlib의 얼굴 검출기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None

    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    face = frame[y:y + h, x:x + w]

    face = cv2.resize(face, (128, 128))
    face = face / 255.0  # Normalize to [0, 1]

    return face

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        face = extract_face(frame)
        if face is not None:
            frames.append(face)
    cap.release()
    return np.array(frames)

def load_image(image_path):
    frame = cv2.imread(image_path)
    face = extract_face(frame)
    return face if face is not None else None

def load_data_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(('.mp4', '.avi', '.mov')):
                data.extend(load_video(path))
            elif filename.endswith(('.jpg', '.jpeg', '.png')):
                face = load_image(path)
                if face is not None:
                    data.append(face)
        except Exception as e:
            logging.warning(f"Failed to process {path}: {e}")
    return np.array(data)

def augment_images(images):
    augmented_images = []
    for img in images:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        augmented_images.append(img)
    return np.array(augmented_images)

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

folder_path = 'data_folder'  # 사용할 폴더 경로
data_frames = load_data_from_folder(folder_path)
logging.info(f"Loaded {len(data_frames)} frames.")

augmented_data_frames = augment_images(data_frames)
logging.info(f"Augmented to {len(augmented_data_frames)} frames.")

dataset = tf.data.Dataset.from_tensor_slices(augmented_data_frames).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

def gradient_penalty(batch_size, real_images, fake_images):
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    real_images = tf.cast(real_images, dtype=tf.float32)
    fake_images = tf.cast(fake_images, dtype=tf.float32)

    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_images_pred = discriminator(interpolated_images, training=True)

    gradients = tape.gradient(interpolated_images_pred, interpolated_images)
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step(image_batch):
    batch_size = tf.shape(image_batch)[0]
    noise = tf.random.normal([batch_size, 100], dtype=tf.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(image_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gp = gradient_penalty(batch_size, image_batch, generated_images)
        disc_loss += tf.cast(gp, dtype=tf.float32)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, output_dir='generated_images'):
    predictions = model(test_input, training=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] * 127.5 + 127.5) / 255.0)
        plt.axis('off')
    output_path = os.path.join(output_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(output_path)
    clear_output(wait=True)
    plt.show()
    plt.close(fig)

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
