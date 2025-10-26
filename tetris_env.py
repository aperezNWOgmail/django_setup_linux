# tetris_env.py
import numpy as np
from typing import Tuple, Dict

# === Constantes ===
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
EMPTY_CELL = 0.0
FILLED_CELL = 1.0

# === Formas de Tetrominós (I, O, T, S, Z, J, L) ===
SHAPES = [
    # I
    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    # O
    [[1, 1],
     [1, 1]],

    # T
    [[0, 1, 0],
     [1, 1, 1],
     [0, 0, 0]],

    # S
    [[0, 1, 1],
     [1, 1, 0],
     [0, 0, 0]],

    # Z
    [[1, 1, 0],
     [0, 1, 1],
     [0, 0, 0]],

    # J
    [[1, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],

    # L
    [[0, 0, 1],
     [1, 1, 1],
     [0, 0, 0]]
]

class TetrisEnv:
    def __init__(self):
        self.board_height = BOARD_HEIGHT
        self.board_width = BOARD_WIDTH
        self.reset()

    def reset(self) -> np.ndarray:
        """Reinicia el entorno."""
        self.board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
        self.score = 0
        self.game_over = False
        self._spawn_piece()
        return self._get_state()

    def _spawn_piece(self) -> None:
        """Genera una nueva pieza aleatoria."""
        shape_idx = np.random.randint(0, len(SHAPES))
        self.current_piece = np.array(SHAPES[shape_idx])
        self.piece_x = self.board_width // 2 - self.current_piece.shape[1] // 2
        self.piece_y = 0
        
        if self._check_collision():
            self.game_over = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una acción.
        Acciones: 0=izq, 1=der, 2=rotar, 3=bajar rápido, 4=nada
        """
        reward = 0.0
        old_board = self.board.copy()

        # Aplicar acción
        if action == 0:  # ← Izquierda
            self.piece_x -= 1
            if self._check_collision(): self.piece_x += 1
        elif action == 1:  # → Derecha
            self.piece_x += 1
            if self._check_collision(): self.piece_x -= 1
        elif action == 2:  # ↑ Rotar
            rotated = list(zip(*self.current_piece[::-1]))
            rotated = np.array(rotated)
            old_piece = self.current_piece
            self.current_piece = rotated
            if self._check_collision(): self.current_piece = old_piece
        elif action == 3:  # ↓ Bajar rápido
            while not self._check_collision(): self.piece_y += 1
            self.piece_y -= 1
            reward += 1.0  # Recompensa por eficiencia
        elif action == 4:  # No hacer nada
            pass

        # Gravedad: intentar bajar
        self.piece_y += 1
        if self._check_collision():
            self.piece_y -= 1
            self._lock_piece()
            lines_cleared = self._clear_lines()
            line_rewards = [0, 100, 300, 500, 800]
            reward += line_rewards[lines_cleared] if lines_cleared < len(line_rewards) else 800
            self.score += line_rewards[lines_cleared] if lines_cleared < len(line_rewards) else 800
            self._spawn_piece()
            if self.game_over:
                reward -= 1.0  # Penalización moderada
        else:
            reward -= 0.01  # Penalización muy baja por tiempo

        next_state = self._get_state()
        done = self.game_over
        info = {"score": float(self.score), "lines": lines_cleared if 'lines_cleared' in locals() else 0}

        return next_state, float(reward), bool(done), info

    def _check_collision(self) -> bool:
        for y in range(self.current_piece.shape[0]):
            for x in range(self.current_piece.shape[1]):
                if self.current_piece[y][x]:
                    px, py = x + self.piece_x, y + self.piece_y
                    if px < 0 or px >= self.board_width or py >= self.board_height:
                        return True
                    if py >= 0 and self.board[py][px] != EMPTY_CELL:
                        return True
        return False

    def _lock_piece(self) -> None:
        for y in range(self.current_piece.shape[0]):
            for x in range(self.current_piece.shape[1]):
                if self.current_piece[y][x]:
                    py, px = y + self.piece_y, x + self.piece_x
                    if 0 <= py < self.board_height and 0 <= px < self.board_width:
                        self.board[py][px] = FILLED_CELL

    def _clear_lines(self) -> int:
        lines = 0
        for y in range(self.board_height - 1, -1, -1):
            if np.sum(self.board[y]) == self.board_width:
                lines += 1
                for yy in range(y, 0, -1):
                    self.board[yy] = self.board[yy - 1].copy()
                self.board[0] = np.zeros(self.board_width)
        return lines

    def _get_state(self) -> np.ndarray:
        return self.board.astype(np.float32)

    def render(self) -> None:
        """Imprime el tablero en consola."""
        print("+" + "-" * self.board_width + "+")
        for row in self.board:
            print("|" + "".join("█" if cell else " " for cell in row) + "|")
        print("+" + "-" * self.board_width + "+")