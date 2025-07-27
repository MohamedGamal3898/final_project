"""
Usage examples for the Django-compatible spatial database service.
This file shows how to migrate from the original DatabaseManager to the new Django implementation.
"""

import os
import django
from django.conf import settings

# Setup Django (if running standalone)
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
    django.setup()

from django.contrib.gis.geos import GEOSGeometry
from spatial_services import SpatialDatabaseService
from models import Facility, Level, Unit, Detail, Landmark, Pathway, Transition


class SpatialDatabaseUsageExample:
    """
    Examples showing how to use the Django-compatible spatial database service.
    These examples replace the functionality from your original DatabaseManager.
    """
    
    def __init__(self):
        self.service = SpatialDatabaseService()
    
    def example_1_create_schema_and_tables(self):
        """
        Example 1: Create schema with tables (equivalent to create_schema_with_tables)
        
        Original code:
        db_manager = DatabaseManager()
        db_manager.create_schema_with_tables("clientschema")
        """
        print("=== Example 1: Creating Schema and Tables ===")
        
        # In Django, tables are created through migrations
        # But we can still create the schema manually
        success = self.service.create_schema_with_tables("clientschema")
        
        if success:
            print("✓ Schema and PostGIS extension created successfully")
            print("Note: Run 'python manage.py migrate' to create tables")
        else:
            print("✗ Failed to create schema")
    
    def example_2_insert_records(self):
        """
        Example 2: Insert records using Django ORM (equivalent to insert_record)
        
        Original code:
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
        """
        print("\n=== Example 2: Inserting Records ===")
        
        # Method 1: Using the service class (similar to original)
        geom_wkt = "POLYGON Z ((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))"
        attribute_dict = {
            'level_id': '123e4567-e89b-12d3-a456-426614174000',
            'name_short': 'L1',
            'elevation_relative': 10.0,
            'facility_id': '123e4567-e89b-12d3-a456-426614174001'  # Required FK
        }
        
        success = self.service.insert_record(
            table_name="levels",
            geom_wkt=geom_wkt,
            schema_name="clientschema",
            attribute_dict=attribute_dict
        )
        
        if success:
            print("✓ Record inserted using service class")
        
        # Method 2: Using Django ORM directly (recommended for Django projects)
        try:
            # First create a facility (required for FK relationship)
            facility = Facility.objects.create(
                facility_id='123e4567-e89b-12d3-a456-426614174001',
                elevation_relative=0.0,
                site_id='123e4567-e89b-12d3-a456-426614174002',
                geom=GEOSGeometry("MULTIPOLYGON Z (((0 0 0, 20 0 0, 20 20 0, 0 20 0, 0 0 0)))", srid=4326)
            )
            
            # Then create the level
            level = Level.objects.create(
                level_id='123e4567-e89b-12d3-a456-426614174000',
                name_short='L1',
                elevation_relative=10.0,
                facility=facility,
                geom=GEOSGeometry(geom_wkt, srid=4326)
            )
            print("✓ Record inserted using Django ORM")
            
        except Exception as e:
            print(f"✗ Error inserting with Django ORM: {e}")
    
    def example_3_fetch_levels_polygons(self):
        """
        Example 3: Fetch level polygons (equivalent to fetch_levels_polygons)
        
        Original code:
        db_manager = DatabaseManager()
        levels = db_manager.fetch_levels_polygons()
        """
        print("\n=== Example 3: Fetching Level Polygons ===")
        
        # Method 1: Using the service class (similar to original)
        levels = self.service.fetch_levels_polygons()
        
        print(f"✓ Found {len(levels)} levels using service class")
        for level_id, level_data in levels.items():
            print(f"  - Level {level_id}: {level_data['name_short']}, Z={level_data['z_value']}")
        
        # Method 2: Using Django ORM directly
        django_levels = Level.objects.select_related('facility').all()
        
        print(f"✓ Found {django_levels.count()} levels using Django ORM")
        for level in django_levels:
            print(f"  - Level {level.level_id}: {level.name_short}")
    
    def example_4_fetch_barriers_polylines(self):
        """
        Example 4: Fetch barrier polylines (equivalent to fetch_barriers_polylines)
        
        Original code:
        barriers = db_manager.fetch_barriers_polylines(
            level_id="123e4567-e89b-12d3-a456-426614174000",
            barriers_list=["wall", "door"]
        )
        """
        print("\n=== Example 4: Fetching Barrier Polylines ===")
        
        level_id = "123e4567-e89b-12d3-a456-426614174000"
        barriers_list = ["wall", "door"]
        
        # Method 1: Using the service class (similar to original)
        barriers = self.service.fetch_barriers_polylines(level_id, barriers_list)
        
        print(f"✓ Found {len(barriers)} barriers using service class")
        
        # Method 2: Using Django ORM directly
        details = Detail.objects.filter(
            level_id=level_id,
            use_type__in=barriers_list
        )
        
        print(f"✓ Found {details.count()} barrier details using Django ORM")
    
    def example_5_create_transitions(self):
        """
        Example 5: Create elevator and stairs transitions
        
        Original code:
        db_manager.create_elevators_transitions()
        db_manager.create_stairs_transitions()
        """
        print("\n=== Example 5: Creating Transitions ===")
        
        # Create elevator transitions
        elevator_success = self.service.create_elevators_transitions()
        if elevator_success:
            print("✓ Elevator transitions created")
        else:
            print("✗ Failed to create elevator transitions")
        
        # Create stairs transitions
        stairs_success = self.service.create_stairs_transitions()
        if stairs_success:
            print("✓ Stairs transitions created")
        else:
            print("✗ Failed to create stairs transitions")
    
    def example_6_bulk_operations(self):
        """
        Example 6: Bulk operations for better performance
        This is a new feature not available in the original DatabaseManager
        """
        print("\n=== Example 6: Bulk Operations ===")
        
        # Prepare bulk data
        bulk_units = [
            {
                'unit_id': f'unit_{i}',
                'elevation_relative': 10.0,
                'level_id': '123e4567-e89b-12d3-a456-426614174000',
                'use_type': 'room',
                'geom_wkt': f"POLYGON Z (({i} {i} 10, {i+1} {i} 10, {i+1} {i+1} 10, {i} {i+1} 10, {i} {i} 10))"
            }
            for i in range(5)
        ]
        
        # Bulk create using service
        success = self.service.bulk_create_records('units', bulk_units)
        
        if success:
            print("✓ Bulk created 5 units")
        else:
            print("✗ Failed to bulk create units")
    
    def example_7_django_orm_features(self):
        """
        Example 7: Advanced Django ORM features for spatial queries
        These features go beyond the original DatabaseManager capabilities
        """
        print("\n=== Example 7: Advanced Django ORM Spatial Queries ===")
        
        from django.contrib.gis.db.models import Q
        from django.contrib.gis.measure import Distance
        
        try:
            # Spatial queries using Django ORM
            point = GEOSGeometry('POINT Z (5 5 10)', srid=4326)
            
            # Find units within distance
            nearby_units = Unit.objects.filter(
                geom__distance_lte=(point, Distance(m=100))
            )
            print(f"✓ Found {nearby_units.count()} units within 100m")
            
            # Find intersecting geometries
            polygon = GEOSGeometry('POLYGON Z ((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))', srid=4326)
            intersecting_units = Unit.objects.filter(geom__intersects=polygon)
            print(f"✓ Found {intersecting_units.count()} intersecting units")
            
            # Complex queries with relationships
            units_on_level_1 = Unit.objects.filter(
                level__name_short='L1',
                use_type__in=['room', 'office']
            ).select_related('level__facility')
            
            print(f"✓ Found {units_on_level_1.count()} rooms/offices on Level 1")
            
        except Exception as e:
            print(f"Note: Spatial queries require data in database: {e}")
    
    def example_8_migration_helper(self):
        """
        Example 8: Helper methods for migrating from original DatabaseManager
        """
        print("\n=== Example 8: Migration Helpers ===")
        
        # Get table metadata (similar to original tables_metadata_provider)
        metadata = self.service.tables_metadata_provider()
        print(f"✓ Found metadata for {len(metadata)} spatial tables")
        
        for schema, table, geom_column in metadata:
            print(f"  - {schema}.{table} ({geom_column})")
        
        # Get Django model by table name
        model_class = self.service.get_model_by_table_name('units')
        if model_class:
            print(f"✓ Found Django model: {model_class.__name__}")


def run_all_examples():
    """Run all usage examples."""
    print("Django Spatial Database Service - Usage Examples")
    print("=" * 60)
    
    try:
        example = SpatialDatabaseUsageExample()
        
        # Run all examples
        example.example_1_create_schema_and_tables()
        example.example_2_insert_records()
        example.example_3_fetch_levels_polygons()
        example.example_4_fetch_barriers_polylines()
        example.example_5_create_transitions()
        example.example_6_bulk_operations()
        example.example_7_django_orm_features()
        example.example_8_migration_helper()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("Make sure Django is properly configured and database is accessible.")


if __name__ == "__main__":
    run_all_examples()