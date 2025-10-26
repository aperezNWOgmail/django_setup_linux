# train_agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import deque
import logging

# === Configuración de logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Parámetros ===
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
INPUT_SHAPE = (BOARD_HEIGHT, BOARD_WIDTH)
NUM_ACTIONS = 5
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.999  # Más lento
TARGET_UPDATE_FREQ = 10
MAX_STEPS_PER_EPISODE = 500

# === Buffer de experiencia ===
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

# === Crear modelo DQN ===
def create_model():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_ACTIONS, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# === Entrenamiento mejorado ===
def train_agent_tetris(env, epochs=500):
    model = create_model()
    target_model = create_model()
    buffer = ReplayBuffer(max_size=REPLAY_BUFFER_SIZE)
    
    epsilon = EPSILON_START
    
    for episode in range(epochs):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            
            # Decidir acción
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                state_batch = np.expand_dims(state, axis=0).astype(np.float32)
                try:
                    q_values = model.predict(state_batch)  # TF 2.8: sin verbose
                except:
                    q_values = model.predict(state_batch)
                action = int(np.argmax(q_values[0]))
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Guardar experiencia
            buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Entrenar si hay suficientes datos
            batch = buffer.sample(BATCH_SIZE)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                
                current_q = model.predict(states)
                next_q = target_model.predict(next_states)
                max_next_q = np.max(next_q, axis=1)
                
                targets = current_q.copy()
                for i in range(BATCH_SIZE):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + GAMMA * max_next_q[i]
                
                history = model.fit(states, targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
                loss = history.history['loss'][0]
            
            if done:
                break
        
        # Actualizar epsilon más lentamente
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)
        
        # Actualizar target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
        
        # Logging mejorado
        logger.info(f"Episode {episode+1}/{epochs} | "
                   f"Score: {env.score} | "
                   f"Epsilon: {epsilon:.3f} | "
                   f"Reward: {total_reward:+.2f} | "
                   f"Steps: {steps} | "
                   f"Buffer: {len(buffer.buffer)}")

    # Guardar modelo
    model.save("tetris_dqn_model.h5")
    logger.info("✅ Modelo guardado como 'tetris_dqn_model.h5'")
    return model