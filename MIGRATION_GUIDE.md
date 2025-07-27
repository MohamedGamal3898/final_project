# Migration Guide: From DatabaseManager to Django GeoDjango

This guide helps you migrate from your original `DatabaseManager` class to the Django-compatible spatial database implementation using GeoDjango.

## Overview

The migration involves:
1. **Models**: Converting database tables to Django models with spatial field support
2. **Services**: Replacing the `DatabaseManager` class with a Django-compatible service
3. **Configuration**: Setting up GeoDjango and spatial database configuration
4. **Dependencies**: Installing required packages for GeoDjango

## Step 1: Install Dependencies

Update your `requirements.txt` to include GeoDjango dependencies:

```txt
Django>=4.2.0
psycopg2-binary>=2.9.0
shapely>=2.0.0
python-dotenv>=1.0.0
```

For spatial functionality, you'll also need system-level dependencies:

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install binutils libproj-dev gdal-bin
sudo apt-get install postgresql-client postgis
```

### macOS (with Homebrew):
```bash
brew install postgresql postgis gdal
```

### Windows:
Use OSGeo4W installer or conda for spatial libraries.

## Step 2: Django Project Setup

If you don't have a Django project yet:

```bash
# Create Django project
django-admin startproject your_project_name
cd your_project_name

# Create an app for your spatial models
python manage.py startapp spatial_app
```

## Step 3: Configure Settings

Update your Django settings based on `django_settings_example.py`:

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.gis',  # Required for GeoDjango
    'spatial_app',         # Your app name
    # ... other apps
]

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': os.getenv('PG_DB'),
        'USER': os.getenv('PG_USER'),
        'PASSWORD': os.getenv('PG_PASS'),
        'HOST': os.getenv('PG_HOST'),
        'PORT': os.getenv('PG_PORT'),
        'OPTIONS': {
            'options': f"-c search_path={os.getenv('DB_SCHEMA', 'clientschema')},public"
        },
    }
}

# Custom spatial settings (from your original DatabaseManager)
SRID = int(os.getenv('SRID', '4326'))
BUFFER_DIST = float(os.getenv('BUFFER_DIST', '0.5'))
DENSIFY_DISTANCE = float(os.getenv('DENSIFY_DISTANCE', '0.1'))
DB_SCHEMA = os.getenv('DB_SCHEMA', 'clientschema')
```

## Step 4: Environment Variables

Update your `.env` file to include Django-specific variables:

```env
# Existing variables (keep these)
SRID=4326
PG_HOST=localhost
PG_PORT=5432
PG_DB=your_database
PG_USER=your_username
PG_PASS=your_password
BUFFER_DIST=0.5
DENSIFY_DISTANCE=0.1
DB_SCHEMA=clientschema

# New Django variables
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
```

## Step 5: Code Migration

### Original DatabaseManager Usage → Django Equivalent

#### 1. Creating Schema and Tables

**Original:**
```python
from .database_manager import DatabaseManager

db_manager = DatabaseManager()
db_manager.create_schema_with_tables("clientschema")
```

**Django:**
```python
# Copy models.py to your app directory
# Copy spatial_services.py to your app directory

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Optional: Create schema manually
from spatial_services import SpatialDatabaseService
service = SpatialDatabaseService()
service.create_schema_with_tables("clientschema")
```

#### 2. Inserting Records

**Original:**
```python
db_manager.insert_record(
    table_name="levels",
    geom_wkt="POLYGON Z ((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))",
    schema_name="clientschema",
    attribute_dict={
        'level_id': '123e4567-e89b-12d3-a456-426614174000',
        'name_short': 'L1',
        'elevation_relative': 10.0
    }
)
```

**Django (Option 1 - Service Class):**
```python
from spatial_services import SpatialDatabaseService
from django.contrib.gis.geos import GEOSGeometry

service = SpatialDatabaseService()
service.insert_record(
    table_name="levels",
    geom_wkt="POLYGON Z ((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))",
    schema_name="clientschema",
    attribute_dict={
        'level_id': '123e4567-e89b-12d3-a456-426614174000',
        'name_short': 'L1',
        'elevation_relative': 10.0,
        'facility_id': 'required_facility_id'  # FK relationship
    }
)
```

**Django (Option 2 - ORM Direct):**
```python
from models import Level, Facility
from django.contrib.gis.geos import GEOSGeometry

level = Level.objects.create(
    level_id='123e4567-e89b-12d3-a456-426614174000',
    name_short='L1',
    elevation_relative=10.0,
    facility=facility_instance,  # FK relationship
    geom=GEOSGeometry("POLYGON Z ((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))", srid=4326)
)
```

#### 3. Fetching Data

**Original:**
```python
levels = db_manager.fetch_levels_polygons()
barriers = db_manager.fetch_barriers_polylines(level_id, ["wall", "door"])
```

**Django:**
```python
# Option 1: Service class (similar interface)
service = SpatialDatabaseService()
levels = service.fetch_levels_polygons()
barriers = service.fetch_barriers_polylines(level_id, ["wall", "door"])

# Option 2: Django ORM (recommended)
from models import Level, Detail

levels = Level.objects.all()
barriers = Detail.objects.filter(
    level_id=level_id,
    use_type__in=["wall", "door"]
)
```

#### 4. Creating Transitions

**Original:**
```python
db_manager.create_elevators_transitions()
db_manager.create_stairs_transitions()
```

**Django:**
```python
service = SpatialDatabaseService()
service.create_elevators_transitions()
service.create_stairs_transitions()
```

## Step 6: Database Migration

### If you have existing data:

1. **Backup your existing database:**
   ```bash
   pg_dump -h localhost -U your_user your_database > backup.sql
   ```

2. **Option A: Keep existing tables and use Django models as views:**
   - Set `managed = False` in model Meta classes
   - Use `db_table` to point to existing tables

3. **Option B: Migrate data to new Django-managed tables:**
   - Export data from existing tables
   - Run Django migrations to create new tables
   - Import data using Django ORM

### Example migration script:

```python
# migrate_data.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
django.setup()

from models import Facility, Level, Unit, Detail, Landmark, Pathway, Transition
from django.contrib.gis.geos import GEOSGeometry
import psycopg2
from psycopg2.extras import RealDictCursor

def migrate_existing_data():
    # Connect to existing database
    conn = psycopg2.connect(
        host=os.getenv('PG_HOST'),
        database=os.getenv('PG_DB'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASS')
    )
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        # Migrate facilities
        cursor.execute("SELECT * FROM clientschema.facility")
        for row in cursor.fetchall():
            facility = Facility(
                facility_id=row['facility_id'],
                elevation_relative=row['elevation_relative'],
                facility_number=row['facility_number'],
                levels_total=row['levels_total'],
                name=row['name'],
                site_id=row['site_id'],
                site_name=row['site_name'],
                geom=GEOSGeometry(row['geom'].wkb, srid=4326)
            )
            facility.save()
        
        # Continue for other tables...

if __name__ == "__main__":
    migrate_existing_data()
```

## Step 7: Testing

Run the usage examples to test your migration:

```bash
python usage_example.py
```

## Step 8: Advanced Features

### New capabilities with Django GeoDjango:

1. **Advanced spatial queries:**
   ```python
   from django.contrib.gis.measure import Distance
   from django.contrib.gis.geos import Point
   
   point = Point(0, 0, srid=4326)
   nearby_units = Unit.objects.filter(
       geom__distance_lte=(point, Distance(m=100))
   )
   ```

2. **Spatial aggregations:**
   ```python
   from django.contrib.gis.db.models import Union
   
   combined_area = Level.objects.aggregate(
       total_area=Union('geom')
   )
   ```

3. **Django Admin integration:**
   ```python
   # admin.py
   from django.contrib.gis import admin
   from models import Level
   
   @admin.register(Level)
   class LevelAdmin(admin.GISModelAdmin):
       gis_widget_kwargs = {
           'attrs': {'default_zoom': 12},
       }
   ```

## Troubleshooting

### Common Issues:

1. **GDAL/GEOS not found:**
   - Install system-level spatial libraries
   - Set library paths in Django settings if needed

2. **SRID mismatches:**
   - Ensure all geometries use the same SRID
   - Use `ST_Transform()` for coordinate system conversions

3. **Foreign key constraint errors:**
   - Django enforces FK relationships strictly
   - Create parent objects before child objects

4. **Migration issues:**
   - Use `--fake-initial` for existing databases
   - Consider custom migration operations for complex schema changes

## Performance Considerations

1. **Use bulk operations:**
   ```python
   service.bulk_create_records('units', bulk_data)
   ```

2. **Optimize queries with select_related/prefetch_related:**
   ```python
   levels = Level.objects.select_related('facility').all()
   ```

3. **Add spatial indexes:**
   ```python
   class Meta:
       db_table = 'levels'
       indexes = [
           models.Index(fields=['geom']),  # Spatial index
       ]
   ```

## Conclusion

This migration provides:
- ✅ Full compatibility with Django ecosystem
- ✅ Better ORM integration and query capabilities
- ✅ Improved performance with bulk operations
- ✅ Enhanced spatial query features
- ✅ Better testing and debugging tools
- ✅ Django Admin integration for data management

The Django version maintains all the functionality of your original `DatabaseManager` while adding the benefits of Django's ORM and GeoDjango's spatial capabilities.