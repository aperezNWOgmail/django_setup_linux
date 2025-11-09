# train_agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import deque
import os
import json

# === Parámetros ajustados para mejor aprendizaje ===
EPISODES = 5000           # Total de episodios
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 20000   # Más grande para recordar experiencias
GAMMA = 0.99               # Factor de descuento
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATE = 0.999  # Más lento: permite explorar más tiempo
TARGET_UPDATE_FREQ = 10
CHECKPOINT_EVERY = 500
MODEL_PATH = 'tetris_dqn_agent.h5'
LOG_FILE = 'training_log.json'

# === Cargar entorno ===
from tetris_env import TetrisEnv

def create_dqn_model():
    """Crea y devuelve el modelo DQN."""
    model = models.Sequential([
        layers.Input(shape=(20, 10)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(5)  # 5 acciones
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# === Buffer de experiencia mejorado ===
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
    
    def size(self):
        return len(self.buffer)

# === Funciones auxiliares para evaluar estado del tablero ===
def get_max_height(board):
    """Altura máxima del stack."""
    if np.any(board.sum(axis=1) > 0):
        return 20 - np.argmax(board.sum(axis=1) > 0)
    return 0

def get_holes(board):
    """Cuenta agujeros (celdas vacías bajo bloques)."""
    holes = 0
    for x in range(10):
        block_found = False
        for y in range(20):
            if board[y][x] == 1:
                block_found = True
            elif block_found and board[y][x] == 0:
                holes += 1
    return holes

def get_bumpiness(board):
    """Diferencia entre columnas adyacentes."""
    heights = [0]*10
    for x in range(10):
        if np.any(board[:, x] == 1):
            heights[x] = 20 - np.argmax(board[:, x] == 1)
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(9))
    return bumpiness

# === Entrenamiento principal ===
def train_dqn():
    env = TetrisEnv()
    model = create_dqn_model()
    target_model = create_dqn_model()
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    epsilon = EPSILON_START
    target_model.set_weights(model.get_weights())
    
    # Registro de métricas
    training_log = []
    
    print("🚀 Iniciando entrenamiento DQN para Tetris...")
    print(f"Episodios: {EPISODES} | Buffer: {REPLAY_BUFFER_SIZE} | Epsilon decay: {EPSILON_DECAY_RATE}")

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        lines_cleared_this_episode = 0
        
        # Estado inicial
        prev_max_height = get_max_height(state)
        prev_holes = get_holes(state)
        
        while steps < 1000 and not env.game_over:
            steps += 1
            
            # Decidir acción (epsilon-greedy)
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                state_batch = np.expand_dims(state, axis=0).astype(np.float32)
                q_values = model.predict(state_batch, verbose=0)
                action = int(np.argmax(q_values[0]))
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            current_score = env.score
            
            # === Mejora clave: Recompensas más densas y útiles ===
            reward = 0.0
            
            # 1. Penalización leve por tiempo
            reward -= 0.01
            
            # 2. Recompensa por bajar (incentiva movimiento útil)
            if action == 3:  # hard drop
                reward += 1.0
            elif action in [0, 1]:  # mover lateralmente
                reward += 0.1
            
            # 3. Recompensa por limpiar líneas (fuerte!)
            new_lines = current_score / 100  # Suponiendo 100 por línea
            if new_lines > lines_cleared_this_episode:
                cleared = int(new_lines - lines_cleared_this_episode)
                reward += [0, 100, 300, 500, 800][cleared] if cleared < 5 else 800
                lines_cleared_this_episode = new_lines
            
            # 4. Penalizaciones por mala forma de juego
            current_max_height = get_max_height(next_state)
            if current_max_height > prev_max_height:
                reward -= 0.5  # Penaliza subir demasiado
            
            current_holes = get_holes(next_state)
            if current_holes > prev_holes:
                reward -= 1.0  # Penaliza crear agujeros
            
            # Actualizar estado anterior
            prev_max_height = current_max_height
            prev_holes = current_holes
            
            # 5. Recompensa final si limpia línea
            if lines_cleared_this_episode > 0:
                reward += lines_cleared_this_episode * 50
            
            # Guardar experiencia
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Entrenar si hay suficientes datos
            batch = buffer.sample(BATCH_SIZE)
            if batch is not None:
                states, actions, rewards, next_states, dones = batch
                
                current_q = model.predict(states, verbose=0)
                next_q = target_model.predict(next_states, verbose=0)
                max_next_q = np.max(next_q, axis=1)
                
                targets = current_q.copy()
                for i in range(BATCH_SIZE):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + GAMMA * max_next_q[i]
                
                model.fit(states, targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
            
            if done:
                break
        
        # Actualizar epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)
        
        # Actualizar target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
        
        # Logging mejorado
        log_entry = {
            "episode": episode + 1,
            "score": env.score,
            "steps": steps,
            "reward": round(total_reward, 2),
            "epsilon": round(epsilon, 3),
            "buffer_size": buffer.size(),
            "lines_cleared": int(lines_cleared_this_episode)
        }
        training_log.append(log_entry)
        
        print(f"[Ep {episode+1:4d}] Score: {env.score:3d} | "
              f"Lines: {int(lines_cleared_this_episode):2d} | "
              f"Steps: {steps:3d} | "
              f"R: {total_reward:+6.2f} | "
              f"ε: {epsilon:.3f} | "
              f"Buf: {buffer.size():5d}")
        
        # Guardar checkpoint
        if (episode + 1) % CHECKPOINT_EVERY == 0 or episode == EPISODES - 1:
            model.save(f'tetris_dqn_agent_ep{episode+1}.h5')
            print(f"💾 Checkpoint guardado: tetris_dqn_agent_ep{episode+1}.h5")
            
            # Guardar log
            with open(LOG_FILE, 'w') as f:
                json.dump(training_log, f, indent=2)
            print(f"📊 Log guardado: {LOG_FILE}")

    # Guardar modelo final
    model.save(MODEL_PATH)
    print(f"✅ Modelo final guardado como '{MODEL_PATH}'")
    return model

# === Ejecutar entrenamiento al llamar al script ===
if __name__ == "__main__":
    train_dqn()