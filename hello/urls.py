from django.urls import path
from hello import views

urlpatterns = [
    path("", views.home, name="home"),
    path("my_view", views.my_view, name="my_view"),
    path('getAllLogs', views.getAllLogs,    name='getAllLogs'),
    path('getAllPersons', views.getAllPersons, name='getAllPersons'),
    path('getAllContactForms', views.getAllContactForms, name='getAllContactForms'),
    path('train_tictactoe_model', views.train_tictactoe_model, name='train_tictactoe_model'),
    path('download_tictactoe_model', views.download_tictactoe_model, name='download_tictactoe_model'),
    path('train_tetris_endpoint', views.train_tetris_endpoint, name='train_tetris_endpoint'),
    path('download_tetris_model', views.download_tetris_model, name='download_tetris_model'),
]