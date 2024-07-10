import cv2
import dlib
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from imutils import face_utils

# 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dlib의 얼굴 검출기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 얼굴 영역 추출 함수
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

# 비디오에서 프레임을 로드하여 얼굴 이미지를 추출하는 함수
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

# GAN 생성자 모델 정의
def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 16 * 16, input_dim=100))
    model.add(tf.keras.layers.Reshape((16, 16, 128)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))

    return model

# GAN 판별자 모델 정의
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(128, 128, 3)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

# GAN 학습을 위한 손실 함수 및 옵티마이저 정의
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 훈련에 사용할 체크포인트 설정
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 데이터셋 준비
BATCH_SIZE = 32
EPOCHS = 100
BUFFER_SIZE = 60000

video_path = '7.mp4'
video_frames = load_video(video_path)
print(f"Loaded {len(video_frames)} frames.")

dataset = tf.data.Dataset.from_tensor_slices(video_frames).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 훈련 함수 정의
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# 훈련 함수 실행
def train(dataset, epochs, checkpoint_dir, save_every=10):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        print(f'Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

        if (epoch + 1) % save_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generate_and_save_images(generator, epoch + 1, seed)

# 이미지 생성 및 저장 함수 정의
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] * 127.5 + 127.5) / 255.0)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# 랜덤 노이즈 벡터 생성
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, 100])

# 학습 중간에 이미지 생성하여 저장
generate_and_save_images(generator, 0, seed)

# 학습 함수 호출
train(dataset, EPOCHS, checkpoint_dir)

# 최종 학습된 모델을 사용하여 이미지 생성
generate_and_save_images(generator, EPOCHS, seed)
