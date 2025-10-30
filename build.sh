#!/usr/bin/env bash
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run migrations and collect static files
python manage.py migrate --noinput
python manage.py collectstatic --noinput

# Optionally create/update a superuser on deploy (controlled by env)
if [ "${AUTO_CREATE_SUPERUSER:-0}" = "1" ] || [ "${AUTO_CREATE_SUPERUSER:-0}" = "true" ]; then
  python scripts/ensure_admin.py || true
fi


