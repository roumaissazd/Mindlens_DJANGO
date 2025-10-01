from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login as auth_login, logout as auth_logout
from .forms import SignUpForm


def home(request):
    return render(request, 'home.html')


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, "Bienvenue sur MindLense !")
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', { 'form': form })


def logout(request):
    """Log out the user via GET and redirect to home."""
    auth_logout(request)
    messages.success(request, "À bientôt ! Vous êtes déconnecté(e).")
    return redirect('home')
