# Generated migration example for your spatial models
# This file should be in your app's migrations/ directory
# Run: python manage.py makemigrations to generate actual migrations

from django.contrib.gis.db import models
from django.db import migrations
from django.conf import settings


def set_geometry_srid(apps, schema_editor):
    """
    Custom migration function to set SRID on geometry fields after table creation.
    This ensures all geometry fields use the SRID from settings.
    """
    db_alias = schema_editor.connection.alias
    
    # Get the SRID from settings
    srid = getattr(settings, 'SRID', 4326)
    
    with schema_editor.connection.cursor() as cursor:
        # Update SRID for all geometry columns
        geometry_updates = [
            f"ALTER TABLE facility ALTER COLUMN geom TYPE geometry(MultiPolygonZ, {srid}) USING ST_SetSRID(geom, {srid});",
            f"ALTER TABLE levels ALTER COLUMN geom TYPE geometry(PolygonZ, {srid}) USING ST_SetSRID(geom, {srid});",
            f"ALTER TABLE units ALTER COLUMN geom TYPE geometry(PolygonZ, {srid}) USING ST_SetSRID(geom, {srid});",
            f"ALTER TABLE details ALTER COLUMN geom TYPE geometry(LineStringZ, {srid}) USING ST_SetSRID(geom, {srid});",
            f"ALTER TABLE landmarks ALTER COLUMN geom TYPE geometry(PointZ, {srid}) USING ST_SetSRID(geom, {srid});",
            f"ALTER TABLE pathways ALTER COLUMN geom TYPE geometry(LineStringZ, {srid}) USING ST_SetSRID(geom, {srid});",
            f"ALTER TABLE transitions ALTER COLUMN geom TYPE geometry(LineStringZ, {srid}) USING ST_SetSRID(geom, {srid});",
        ]
        
        for update_sql in geometry_updates:
            try:
                cursor.execute(update_sql)
            except Exception as e:
                print(f"Warning: Could not update SRID - {e}")


def create_schema(apps, schema_editor):
    """
    Create the custom schema if it doesn't exist.
    """
    schema_name = getattr(settings, 'DB_SCHEMA', 'clientschema')
    
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;")


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        # Add any dependencies here if needed
    ]

    operations = [
        # Create schema and enable PostGIS
        migrations.RunPython(create_schema, reverse_code=migrations.RunPython.noop),
        
        # Create Facility model
        migrations.CreateModel(
            name='Facility',
            fields=[
                ('elevation_relative', models.FloatField()),
                ('facility_number', models.IntegerField(blank=True, null=True)),
                ('levels_total', models.IntegerField(blank=True, null=True)),
                ('facility_id', models.UUIDField(primary_key=True, serialize=False)),
                ('name', models.TextField(blank=True, null=True)),
                ('site_id', models.UUIDField()),
                ('site_name', models.TextField(blank=True, null=True)),
                ('geom', models.MultiPolygonField(dim=3, srid=4326)),  # Will be updated by custom function
            ],
            options={
                'db_table': 'facility',
            },
        ),
        
        # Create Level model
        migrations.CreateModel(
            name='Level',
            fields=[
                ('elevation_relative', models.FloatField()),
                ('level_id', models.UUIDField(primary_key=True, serialize=False)),
                ('level_number', models.IntegerField(blank=True, null=True)),
                ('access_type', models.TextField(blank=True, null=True)),
                ('name', models.TextField(blank=True, null=True)),
                ('name_short', models.TextField()),
                ('geom', models.PolygonField(dim=3, srid=4326)),  # Will be updated by custom function
                ('facility', models.ForeignKey(
                    db_column='facility_id',
                    on_delete=models.CASCADE,
                    to='your_app.facility',  # Replace 'your_app' with actual app name
                    to_field='facility_id'
                )),
            ],
            options={
                'db_table': 'levels',
            },
        ),
        
        # Create Unit model
        migrations.CreateModel(
            name='Unit',
            fields=[
                ('elevation_relative', models.FloatField()),
                ('name', models.TextField(blank=True, null=True)),
                ('unit_id', models.TextField(primary_key=True, serialize=False)),
                ('use_type', models.TextField(blank=True, null=True)),
                ('geom', models.PolygonField(dim=3, srid=4326)),  # Will be updated by custom function
                ('level', models.ForeignKey(
                    db_column='level_id',
                    on_delete=models.CASCADE,
                    to='your_app.level',  # Replace 'your_app' with actual app name
                    to_field='level_id'
                )),
            ],
            options={
                'db_table': 'units',
            },
        ),
        
        # Create Detail model
        migrations.CreateModel(
            name='Detail',
            fields=[
                ('elevation_relative', models.FloatField()),
                ('use_type', models.TextField(blank=True, null=True)),
                ('detail_id', models.TextField(primary_key=True, serialize=False)),
                ('ocls_extruded_base', models.IntegerField(blank=True, null=True)),
                ('olcs_extruded_height', models.IntegerField(blank=True, null=True)),
                ('geom', models.LineStringField(dim=3, srid=4326)),  # Will be updated by custom function
                ('level', models.ForeignKey(
                    db_column='level_id',
                    on_delete=models.CASCADE,
                    to='your_app.level',  # Replace 'your_app' with actual app name
                    to_field='level_id'
                )),
            ],
            options={
                'db_table': 'details',
            },
        ),
        
        # Create Landmark model
        migrations.CreateModel(
            name='Landmark',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('elevation_relative', models.FloatField()),
                ('name', models.TextField(blank=True, null=True)),
                ('unit_id', models.TextField()),
                ('use_type', models.TextField(blank=True, null=True)),
                ('geom', models.PointField(dim=3, srid=4326)),  # Will be updated by custom function
                ('level', models.ForeignKey(
                    db_column='level_id',
                    on_delete=models.CASCADE,
                    to='your_app.level',  # Replace 'your_app' with actual app name
                    to_field='level_id'
                )),
            ],
            options={
                'db_table': 'landmarks',
            },
        ),
        
        # Create Pathway model
        migrations.CreateModel(
            name='Pathway',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('geom', models.LineStringField(dim=3, srid=4326)),  # Will be updated by custom function
                ('level', models.ForeignKey(
                    db_column='level_id',
                    on_delete=models.CASCADE,
                    to='your_app.level',  # Replace 'your_app' with actual app name
                    to_field='level_id'
                )),
            ],
            options={
                'db_table': 'pathways',
            },
        ),
        
        # Create Transition model
        migrations.CreateModel(
            name='Transition',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('elevator_id', models.IntegerField(blank=True, null=True)),
                ('unit_id', models.TextField(blank=True, null=True)),
                ('use_type', models.TextField(blank=True, null=True)),
                ('geom', models.LineStringField(dim=3, srid=4326)),  # Will be updated by custom function
            ],
            options={
                'db_table': 'transitions',
            },
        ),
        
        # Set proper SRID for all geometry fields based on settings
        migrations.RunPython(set_geometry_srid, reverse_code=migrations.RunPython.noop),
    ]