from django.http import FileResponse, JsonResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import serializers
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
import shutil
import tempfile
import random

##############################################333
# BEGIN TETRIS FUNCIONALITY
##############################################333

class TetrisEnv:
    def __init__(self):
        self.width = 10
        self.height = 20
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.game_over = False
        self.current_piece = self._get_random_piece()
        self.piece_x = self.width // 2 - 2
        self.piece_y = 0

    def _get_random_piece(self):
        # Return a random Tetromino shape (e.g., I, O, T, S, Z, J, L)
        return random.choice([
            [[1, 1, 1, 1]],  # I
            [[1, 1], [1, 1]],  # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 1, 0], [0, 1, 1]],  # Z
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]]   # L
        ])

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.game_over = False
        self.current_piece = self._get_random_piece()
        self.piece_x = self.width // 2 - 2
        self.piece_y = 0
        return self._get_state()

    def _get_state(self):
        # Flatten the board (10x20 = 200 cells)
        flat_board = [cell for row in self.board for cell in row]

        # Flatten the current piece (up to 16 cells, padded/truncated to 16 cells)
        flat_piece = [cell for row in self.current_piece for cell in row]
        flat_piece = flat_piece[:16] + [0] * (16 - len(flat_piece))  # Ensure 16 cells

        # Add piece position (2 cells)
        return flat_board + flat_piece + [self.piece_x, self.piece_y]  # Total: 200 + 16 + 2 = 218

    def step(self, action):
        # Actions: 0 = left, 1 = right, 2 = rotate, 3 = down
        if action == 0:
            self._move_piece(-1, 0)
        elif action == 1:
            self._move_piece(1, 0)
        elif action == 2:
            self._rotate_piece()
        elif action == 3:
            self._move_piece(0, 1)

        # Check if the piece can move down
        if not self._move_piece(0, 1):
            self._lock_piece()
            self._clear_lines()
            self.current_piece = self._get_random_piece()
            self.piece_x = self.width // 2 - 2
            self.piece_y = 0
            if self._check_collision(0, 0):
                self.game_over = True

        reward = self._calculate_reward()
        return self._get_state(), reward, self.game_over

    def _move_piece(self, dx, dy):
        if not self._check_collision(dx, dy):
            self.piece_x += dx
            self.piece_y += dy
            return True
        return False

    def _rotate_piece(self):
        rotated = list(zip(*self.current_piece[::-1]))
        if not self._check_collision(0, 0, rotated):
            self.current_piece = rotated

    def _check_collision(self, dx=0, dy=0, piece=None):
        piece = piece or self.current_piece
        for i, row in enumerate(piece):
            for j, cell in enumerate(row):
                if cell:
                    x = self.piece_x + j + dx
                    y = self.piece_y + i + dy
                    if x < 0 or x >= self.width or y >= self.height or (y >= 0 and self.board[y][x]):
                        return True
        return False

    def _lock_piece(self):
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    self.board[self.piece_y + i][self.piece_x + j] = 1

    def _clear_lines(self):
        new_board = [row for row in self.board if 0 in row]
        lines_cleared = len(self.board) - len(new_board)
        self.score += lines_cleared ** 2
        self.board = [[0] * self.width for _ in range(lines_cleared)] + new_board

    def _calculate_reward(self):
        # Reward based on lines cleared and survival
        return self.score + 1  # Add a small constant to encourage survival

##########################

def build_model_tetris(input_shape, num_actions):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Update input_shape to match the new state size
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

#############################################


def train_tetris(env, model, episodes=1000, max_steps=500, gamma=0.99):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:

            epsilon = max(0.01, 1.0 - episode / episodes)  # Decay epsilon over time
            if random.random() < epsilon:
                action = random.randint(0, 218 - 1)  # Random action
            else:
                action_probs = model.predict(np.array([state]), verbose=0)[0]
                action = np.argmax(action_probs)  # Greedy action

            # Take action and observe next state
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Prepare training data
            target = reward
            if not done:
                next_action_probs = model.predict(np.array([next_state]), verbose=0)[0]
                target += gamma * np.max(next_action_probs)

            target_vec = np.zeros(len(action_probs))
            target_vec[action] = target

            # Train the model
            model.fit(np.array([state]), np.array([target_vec]), epochs=1, verbose=0)

            state = next_state
            step += 1

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

##############################################333

def train_tetris_endpoint(request):
    # Initialize environment and model
    env   = TetrisEnv()
    state = env.reset()
    print(f"State size: {len(state)}")  # Should match the expected input size (e.g., 208)

    # Example: If the state size is now 210
    model = build_model_tetris(input_shape=218, num_actions=4)

    # Train the model
    train_tetris(env, model)

    model.save("tetris_model.h5")

    return HttpResponse("train_tetris_endpoint")

##############################################333
# END TETRIS FUNCIONALITY
##############################################333


def home(request):
    return HttpResponse("Hello, Django!")


def my_view(request):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT TOP 10 * FROM accessLogs order by ID_COLUMN DESC")
        rows = cursor.fetchall()

    # Process rows or pass them to the template context
    context = {'rows': rows}
    return render(request, 'my_template.html', context)

# Define a serializer to handle the raw data.
class RawDataSerializer(serializers.Serializer):
    def to_representation(self, instance):
        # Convert row tuple to dictionary
        fields = [field[0] for field in self.context['cursor'].description]
        return dict(zip(fields, instance))


@api_view(['GET'])
def getAllLogs(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                 SELECT TOP 100
                       AL.[ID_column]  id_Column
                      ,AL.[PageName]   pageName
                      ,AL.[AccessDate] accessDate
                      ,AL.[IpValue]    ipValue
                 FROM
                       [dbo].[accessLogs] AL
                    WHERE
                       AL.[LogType] = 1
                    AND
                        (AL.PAGENAME LIKE '%DEMO%'
                    and
                        AL.PAGENAME LIKE '%PAGE%')
                    AND
                        AL.PAGENAME NOT LIKE '%ERROR%'
                    AND
                        AL.PAGENAME  NOT LIKE '%PAGE_DEMO_INDEX%'
                    AND
                        UPPER(AL.PAGENAME) NOT LIKE '%CACHE%'
                    AND
                        AL.IPVALUE <> '::1'
                 order by
                       AL.[ID_column] desc
                  """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})   # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def getAllPersons(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                  SELECT
                     [Id_Column]        id_Column
                    ,[NombreCompleto]   nombreCompleto
                    ,[ProfesionOficio]  profesionOficio
                    ,[Ciudad]           ciudad
                FROM
                    [dbo].[Persona]
                ORDER BY
                    Id_Column

                """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def getAllContactForms(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                  SELECT
                        id         id
                        ,Name      name
                        ,Email     field_1
                        ,Message   field_2
                        ,CreatedAt field_3
                FROM
                    ContactForm
                ORDER BY
                    id desc
                """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

# === Generate synthetic self-play data ===


def generate_game():
    board = np.zeros(9, dtype=int)
    moves = []
    turn = 1
    while True:
        empty = np.where(board == 0)[0]
        if len(empty) == 0:
            break
        move = np.random.choice(empty)
        moves.append((board.copy(), move))
        board[move] = turn

        b = board.reshape(3, 3)
        win_patterns = [
            b[0, :], b[1, :], b[2, :],
            b[:, 0], b[:, 1], b[:, 2],
            [b[0, 0], b[1, 1], b[2, 2]],
            [b[0, 2], b[1, 1], b[2, 0]]
        ]
        won = False
        for pattern in win_patterns:
            if all(p == turn for p in pattern):
                won = True
                break
        if won:
            break
        turn = -turn
    return [(state.flatten(), move) for state, move in moves]


@api_view(['GET'])
def train_tictactoe_model(request):
    try:
        # Generate dataset
        X_train = []
        y_train = []
        for _ in range(3000):
            game = generate_game()
            for state, move in game:
                X_train.append(state)
                y_train.append(move)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # ✅ Save model in SavedModel format (for C++ compatibility)
        save_dir = "tictactoe_tf_model"
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)  # Remove old model

        model.save(save_dir, save_format="tf")  # ← This was missing!

        return JsonResponse({
            'status': 'success',
            'message': 'Model trained and saved!',
            'save_path': save_dir
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
def download_tictactoe_model(request):
    model_dir = "tictactoe_tf_model"
    if not os.path.exists(model_dir):
        return JsonResponse({"error": "Model not trained yet."}, status=404)

    try:
        # Create a temporary directory for the zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'tictactoe_tf_model.zip')

        # Create ZIP archive
        shutil.make_archive(
            # output path (no extension)
            base_name=os.path.join(temp_dir, 'tictactoe_tf_model'),
            format='zip',
            root_dir=model_dir  # directory to compress
        )

        # Serve the file
        response = FileResponse(
            open(zip_path, 'rb'),
            content_type='application/zip'
        )
        response['Content-Disposition'] = 'attachment; filename="tictactoe_tf_model.zip"'
        return response

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
