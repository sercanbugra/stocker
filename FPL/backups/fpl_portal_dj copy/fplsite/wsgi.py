"""
WSGI config for fplsite project.
This exposes the WSGI callable as a module-level variable named `application`.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fplsite.settings")

application = get_wsgi_application()
