# IMPORTS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.commom.action_chains import ActionChains
import os
from time import sleep
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from collections import deque

# GLOBAL VARIABLES
dino_path = "chrome://dino"
images_dir = os.path.join(os.getcwd(), "dino_images")

# Functions
def preprocess_image(image_path, save_path):
    image = cv2.imread(image_path)
    cv2.waitKey(0)

    # Height
    assert image.shape[0] >= 600
    # width
    assert image.shape[1] >= 1200

    # Crop to 1200x600
    # y and x are flipped
    image = image[:600, :1200]

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Finding Canny Edges
    edged = cv2.Canny(gray, 30, 200)
    cv2.waitKey(0)


    # Rescale - (width, height)
    edged = cv2.resize(edged, (120, 60),
        interpolation = cv2.INTER_AREA)

    # Optional
    # cv2.imwrite(save_path, edged)

    return edged

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) *
                        discount_factor * max_next_Q_values)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optmizer.apply_gradients(zip(grads, model.trainable_variables))

# Classes
class DinoEnv:
    def __init__(self):
        self.driver = None
        self.body = None

    def reset(self):
        if self.driver != None:
            self.driver.send_keys(Keys.UP)
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1600,900")
        self.driver = webdriver.Chrome(executable_path="chromedriver.exe",
                        options=chrome_options)
        try:
            self.driver.get(dino_path)
        except Exception as err:
            print(err)

        self.driver.execute_script("Runner.prototype.onVisibilityChange = function(){};")

        self.body = self.driver.find_element_by_tag_name("body")
        self.body.send_keys(Keys.UP)

        sleep(2)
        # self.pause()

    def step(self, action):
        # self.play()
        print(action)
        if action == 0:
            pass
        elif action == 1:
            self.body.send_keys(Keys.UP)
        elif action == 2:
            self.action_chain.keyDown(Keys.DOWN)
        sleep(0.1)
        if action == 2:
            ActionChains(self.driver).keyUp(Keys.DOWN).perform()
        next_state = self.get_state()
        done = self.get_done()
        if done:
            reward = 1
        else:
            reward = -10
        # self.pause()
        return next_state, reward, done

    def get_state(self):
        screenshot_path = os.path.join(images_dir, "screenshot.png")
        self.driver.save_screenshot(screenshot_path)
        state = preprocess_image(screenshot_path, os.path.join(images_dir, "processed.png"))
        return state

    def get_done(self):
        done = not self.driver.execute_script("return Runner.instance_.playing")
        return done

    def pause(self):
        self.driver.execute_script("Runner.instance_.playing = false")

    def play(self):
        self.driver.execute_script("Runner.instance_.playing = true")

env = DinoEnv()
n_outputs = 3

model = keras.Sequential([
    Conv2D(64, 5, padding="same", input_shape=(60, 120, 1)),
    BatchNormalization(),
    Conv2D(64, 5),
    BatchNormalization(),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Conv2D(128, 5),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 5),
    BatchNormalization(),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_outputs, activation="softmax")
])

print(model.summary())

replay_buffer = deque(maxlen=2000)


batch_size = 32
discount_factor = 0.9
optmizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

for episode in range(600):
    obs = env.reset()
    print(f"Episode: {episode}")
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        training_step(batch_size)
