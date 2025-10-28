from django.urls import path
from django.contrib.auth import views as auth_views
from . import views


urlpatterns = [
    # Home and auth
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', views.logout, name='logout'),

    # Notes CRUD
    path('notes/', views.note_list, name='note_list'),
    path('notes/create/', views.note_create, name='note_create'),
    path('notes/<int:pk>/', views.note_detail, name='note_detail'),
    path('notes/<int:pk>/edit/', views.note_edit, name='note_edit'),
    path('notes/<int:pk>/delete/', views.note_delete, name='note_delete'),
    path('notes/<int:pk>/toggle-favorite/', views.note_toggle_favorite, name='note_toggle_favorite'),

    # Search and dashboard
    path('notes/search/', views.note_search, name='note_search'),
    path('dashboard/', views.dashboard, name='dashboard'),

    # Export
    path('notes/export/json/', views.note_export_json, name='note_export_json'),

    # Profile
    path('profile/', views.profile_view, name='profile'),

    # Détection de visages
    path('notes/<int:pk>/detect-faces/', views.detect_faces_in_note, name='detect_faces_in_note'),
]


