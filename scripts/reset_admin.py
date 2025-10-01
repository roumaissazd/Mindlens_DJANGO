import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mindlens_proj.settings")
import django
django.setup()
from django.contrib.auth.models import User

username = "ikram"
email = "ikram@gmail.com"
password = "admin123"

user, _ = User.objects.get_or_create(username=username, defaults={"email": email})
user.email = email
user.is_staff = True
user.is_superuser = True
user.set_password(password)
user.save()
print("RESET_OK")



