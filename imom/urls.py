from django.urls import path
from . import views

app_name = 'imom'
urlpatterns = [
    path('home', views.Home.as_view(), name='home'),
    path('transcript/<int:mid>', views.transcript, name='transcript'),
    path('summary/<int:mid>', views.summary, name='summary'),
    path('delete/<int:mid>', views.delete, name='delete'),
    path('dwnld_summary/<int:mid>', views.download_summary, name='dwnld_summary'),
    path('dwnld_transcript/<int:mid>', views.download_transcript, name='dwnld_transcript'),
]