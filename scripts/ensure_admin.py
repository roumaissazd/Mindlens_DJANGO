import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mindlens_proj.settings")

import django  # noqa: E402

django.setup()

from django.contrib.auth.models import User  # noqa: E402


def main() -> None:
    auto_flag = os.environ.get("AUTO_CREATE_SUPERUSER", "0").strip()
    if auto_flag not in {"1", "true", "True", "YES", "yes"}:
        return

    username = os.environ.get("DJANGO_SUPERUSER_USERNAME", "admin").strip()
    email = os.environ.get("DJANGO_SUPERUSER_EMAIL", "admin@example.com").strip()
    password = os.environ.get("DJANGO_SUPERUSER_PASSWORD", "").strip()

    if not password:
        # Do not proceed without an explicit password for security
        print("SKIP_SUPERUSER: missing DJANGO_SUPERUSER_PASSWORD")
        return

    user, _created = User.objects.get_or_create(
        username=username, defaults={"email": email}
    )
    user.email = email
    user.is_staff = True
    user.is_superuser = True
    user.set_password(password)
    user.save()
    print(f"SUPERUSER_OK: username={username}")


if __name__ == "__main__":
    main()


