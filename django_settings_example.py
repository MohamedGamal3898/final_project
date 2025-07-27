"""
Django settings for spatial database project.
This file shows the required settings for your GeoDjango project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'your-secret-key-here')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',  # Required for GeoDjango
    # Your app here (replace 'your_app_name' with actual app name)
    # 'your_app_name',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'your_project.urls'  # Replace with your project name

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'your_project.wsgi.application'  # Replace with your project name

# Database configuration for PostGIS
DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': os.getenv('PG_DB', 'your_database_name'),
        'USER': os.getenv('PG_USER', 'your_username'),
        'PASSWORD': os.getenv('PG_PASS', 'your_password'),
        'HOST': os.getenv('PG_HOST', 'localhost'),
        'PORT': os.getenv('PG_PORT', '5432'),
        'OPTIONS': {
            'options': f"-c search_path={os.getenv('DB_SCHEMA', 'clientschema')},public"
        },
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Custom settings from your original DatabaseManager
SRID = int(os.getenv('SRID', '4326'))  # Default to WGS84 if not specified
BUFFER_DIST = float(os.getenv('BUFFER_DIST', '0.5'))
DENSIFY_DISTANCE = float(os.getenv('DENSIFY_DISTANCE', '0.1'))
DB_SCHEMA = os.getenv('DB_SCHEMA', 'clientschema')

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'spatial_operations.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'your_app_name': {  # Replace with your actual app name
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# GeoDjango specific settings
# If you're using GDAL/GEOS libraries in a non-standard location, specify them here:
# GDAL_LIBRARY_PATH = '/path/to/gdal/library'
# GEOS_LIBRARY_PATH = '/path/to/geos/library'
# PROJ4_LIBRARY_PATH = '/path/to/proj4/library'

# For development, you might want to set these if libraries are in custom locations
# On Ubuntu/Debian, these are typically not needed as packages install to standard locations
# On Windows or macOS, you might need to set these paths