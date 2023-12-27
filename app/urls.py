# yourapp/urls.py
from django.urls import path
from .views import analyse_tweet

urlpatterns = [
    path('', analyse_tweet, name='analyse_tweet'),
]