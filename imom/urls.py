from django.urls import path
from . import views

app_name = 'imom'
urlpatterns = [
    path('home', views.Home.as_view(), name='home'),
    path('transcript/<int:mid>', views.transcript, name='transcript'),
    path('delete/<int:mid>', views.delete, name='delete')
]