from django.http import FileResponse, JsonResponse,  HttpResponse
from django.shortcuts import render
from django.db import connection
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import serializers
from tensorflow.keras import layers, models
from typing import Tuple, Dict
import json
import zipfile
import shutil
import tempfile
import random
import numpy as np
import tensorflow as tf
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings


# 333
# BEGIN TETRIS FUNCIONALITY
# 333


#############################################
# === Ruta al modelo ===
MODEL_PATH = os.path.join(settings.BASE_DIR, '', 'tetris_dqn_agent.h5')

# === Cargar modelo al inicio (una sola vez) ===
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            # === Crear y guardar el modelo ===
            _model = tf.keras.models.load_model('tetris_dqn_agent.h5')
            print(f"✅ Modelo cargado desde: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
    return _model

# === Vista: POST /api/tetris/move/ ===
@method_decorator(csrf_exempt, name='dispatch')
class GetAIMoveView(View):
    
    def get_tetris_features(board):
        """board: 20x10 numpy array"""
        if board.sum() == 0:
            return [0.0, 0.0, 0.0, 0.0]

        # 1. Max height
        filled_rows = np.where(board.sum(axis=1) > 0)[0]
        max_height = float(20 - filled_rows[0]) if len(filled_rows) > 0 else 0.0

        # 2. Number of holes
        cumulative = np.cumsum(board, axis=0)
        holes = ((cumulative > 0) & (board == 0)).sum()

        # 3. Bumpiness (sum of absolute differences between adjacent columns)
        heights = []
        for col in range(10):
            if np.any(board[:, col]):
                h = 20 - np.argmax(board[:, col])
            else:
                h = 0
            heights.append(h)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(9))

        # 4. Lines cleared (potential)
        lines_cleared = float(np.sum(board.sum(axis=1) == 10))

        return [max_height, float(holes), float(bumpiness), lines_cleared]

        # Example usage
        board = np.zeros((20, 10))  # Your current board
        features = np.array([get_tetris_features(board)], dtype=np.float32)

        q_values = model.predict(features, verbose=0)[0]
        action = int(np.argmax(q_values))
        print("Suggested action:", action)  # 0=left, 1=right, 2=rotate, 3=drop, 4=nothing
        # === En tu post() ===
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            board = data.get("board")
            if not board:
                return JsonResponse({"error": "Falta 'board'"}, status=400)

            board_array = np.array(board, dtype=np.float32)
            if board_array.shape != (20, 10):
                return JsonResponse({"error": "Debe ser 20x10"}, status=400)

            model = load_model()
            if model is None:
                return JsonResponse({"error": "Modelo no disponible"}, status=500)

            # 🔁 Extrae las 4 características
            features = np.array([get_tetris_features(board_array)], dtype=np.float32)

            # ✅ Ahora sí: predice con forma (1, 4)
            q_values = model.predict(features, verbose=0)[0]
            action = int(np.argmax(q_values))

            action_names = ["move_left", "move_right", "rotate", "hard_drop", "no_action"]

            return JsonResponse({
                "action": action,
                "action_name": action_names[action],
                "q_values": q_values.tolist(),
                "success": True
            })

        except Exception as e:
            return JsonResponse({"error": f"Error interno: {str(e)}"}, status=500)
            
# ###########################################
# END TETRIS FUNCIONALITY
# ###########################################

# ###########################################
# BEGIN TIC TAC TOE FUNCIONALITY
# ###########################################


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

# ######################################
# END  TIC TAC TOE FUNCIONALITY
# ######################################


# #######################################
# BEGIN DATABASE END POINTS
# #######################################

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


# ##########################
# END DATABASE END POINTS
# ##########################
