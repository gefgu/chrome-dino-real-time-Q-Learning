# IMPORTS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
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
    assert image.shape[0] >= 500
    # width
    assert image.shape[1] >= 1000

    # Crop to 1000x500
    # y and x are flipped
    image = image[50:450, :1000]

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Finding Canny Edges
    edged = cv2.Canny(gray, 30, 200)
    cv2.waitKey(0)


    # Rescale - (width, height)
    edged = cv2.resize(edged, (60, 60),
        interpolation = cv2.INTER_AREA)

    # Transform on black and white
    edged[edged > 20] = 255

    # Optional
    cv2.imwrite(save_path, edged)

    edged = edged.reshape(60, 60, 1)
    edged = edged / 255.0

    return edged

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        print("Random: ", end="")
        return np.random.randint(n_outputs)
    else:
        print("Model's Prediction: ", end="")
        Q_values = model.predict(state.reshape(-1, 60, 60, 1))
        return np.argmax(Q_values)

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
        if self.driver is not None:
            self.body.send_keys(Keys.UP)
        else:
            chrome_options = Options()
            chrome_options.add_argument("disable-infobars")
            chrome_options.add_argument("--window-size=1200,675")
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
        return self.get_state()

    def step(self, action):
        # self.play()
        print(action)
        if action == 1:
            pass
        elif action == 2:
            self.body.send_keys(Keys.UP)
        elif action == 0:
            ActionChains(self.driver).key_down(Keys.DOWN).perform()
            # self.body.send_keys(Keys.DOWN)
        sleep(0.1)
        if action == 0:
            ActionChains(self.driver).key_up(Keys.DOWN).perform()
        # self.pause()
        next_state = self.get_state()
        done = self.get_done()
        reward = self.get_reward(done)
        return next_state, reward, done

    def get_state(self):
        screenshot_path = os.path.join(images_dir, "screenshot.png")
        self.driver.save_screenshot(screenshot_path)
        state = preprocess_image(screenshot_path, os.path.join(images_dir, "processed.png"))
        return state

    def get_done(self):
        done = not self.driver.execute_script("return Runner.instance_.playing")
        return done

    def get_reward(self, done):
        score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        score = int(score)
        if done:
            return -1 * score / 10
        else:
            return 0.1 * score / 10

    def pause(self):
        self.driver.execute_script("Runner.instance_.stop();")

    def play(self):
        self.driver.execute_script("Runner.instance_.play();")

env = DinoEnv()
n_outputs = 3

model = keras.Sequential([
    Conv2D(16, 8, strides=4, input_shape=(60, 60, 1), activation="elu"),
    Conv2D(32, 4, strides=2, activation="elu"),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(n_outputs)
])

replay_buffer = deque(maxlen=50000)


batch_size = 32
discount_factor = 0.95
optmizer = keras.optimizers.RMSprop(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

for episode in range(10000):
    sleep(1)
    obs = env.reset()
    print(f"Episode: {episode}")
    while True:
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode >= 50:
        training_step(batch_size)
# env.reset()
# env.get_state()
