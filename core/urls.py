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



  # Resume
    path('resumes/', views.resume_list, name='resume_list'),
    path('resumes/<int:pk>/', views.resume_detail, name='resume_detail'),
    path('resumes/generate/', views.resume_generate, name='resume_generate'),
    path('resume/<int:pk>/edit/', views.resume_edit, name='resume_edit'),
    path('resumes/<int:pk>/delete/', views.resume_delete, name='resume_delete'),  # FIXED
]


