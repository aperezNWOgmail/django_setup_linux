from django.http import FileResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.db import connection
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import serializers
import numpy as np
import tensorflow as tf
import json
import os
import shutil
import tempfile

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
                                           'cursor': cursor})  # Pass cursor for field names
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

@api_view(['GET'])
def train_tictactoe_model(request):
    #if request.method != 'POST':
    #    return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        # === STEP 1: Generate synthetic self-play data ===
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

                # Check win (rows, cols, diag)
                b = board.reshape(3, 3)
                win_patterns = [
                    b[0,:], b[1,:], b[2,:],           # rows
                    b[:,0], b[:,1], b[:,2],           # cols
                    [b[0,0], b[1,1], b[2,2]],         # diag \
                    [b[0,2], b[1,1], b[2,0]]          # diag /
                ]
                for pattern in win_patterns:
                    if all(p == turn for p in pattern):
                        break
                else:
                    turn = -turn
                    continue
                break
            return [(b.flatten(), m) for b, m in moves]

        # Generate dataset
        X_train = []
        y_train = []
        for _ in range(3000):  # ~27k moves
            game = generate_game()
            for state, move in game:
                X_train.append(state)
                y_train.append(move)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

        # === STEP 2: Build & Train Model ===
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

        history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # === STEP 3: Save Model ===
        save_dir = "models/tictactoe_tf_model"
        os.makedirs(save_dir, exist_ok=True)
        #model.save(save_dir)
        model.export("tictactoe_tf_model")
        

        return JsonResponse({
            'status': 'success',
            'message': 'Model trained and saved!',
            'epochs': 10,
            'dataset_size': len(X_train),
            'save_path': save_dir
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

import os
import shutil
import tempfile
from django.http import FileResponse, JsonResponse
import zipfile

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
            base_name=os.path.join(temp_dir, 'tictactoe_tf_model'),  # output path (no extension)
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