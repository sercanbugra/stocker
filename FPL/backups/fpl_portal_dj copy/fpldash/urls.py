from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/myteam", views.api_myteam, name="api_myteam"),
    path("api/data", views.api_data, name="api_data"),
    path("api/suggestions", views.api_suggestions, name="api_suggestions"),
]
