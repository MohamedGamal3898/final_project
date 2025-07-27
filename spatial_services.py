import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from django.conf import settings
from django.db import connection, transaction
from django.contrib.gis.geos import GEOSGeometry, LineString, Point
from django.contrib.gis.db.models import Q
from django.core.exceptions import ValidationError
from shapely import wkb
from shapely.geometry import LineString as ShapelyLineString
from .models import Facility, Level, Unit, Detail, Landmark, Pathway, Transition

logger = logging.getLogger(__name__)


class SpatialDatabaseService:
    """
    Django-compatible service for spatial database operations.
    Replaces the original DatabaseManager with Django ORM integration.
    """
    
    def __init__(self):
        self.srid = getattr(settings, 'SRID', 4326)
        self.buffer_dist = float(getattr(settings, 'BUFFER_DIST', 0.5))
        self.densify_distance = float(getattr(settings, 'DENSIFY_DISTANCE', 0.1))
        self.db_schema = getattr(settings, 'DB_SCHEMA', 'clientschema')
        self.logger = logger

    def create_schema_with_tables(self, schema_name: str = None) -> bool:
        """
        Creates schema and tables using Django migrations.
        In Django, this is typically handled by migrations, but we can create schema manually.
        """
        schema_name = schema_name or self.db_schema
        
        try:
            with connection.cursor() as cursor:
                # Create schema if it doesn't exist
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
                
                # Enable PostGIS extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public")
                
                self.logger.info(f"Schema '{schema_name}' created successfully with PostGIS enabled.")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create schema '{schema_name}': {e}")
            return False

    @transaction.atomic
    def create_elevators_transitions(self) -> bool:
        """
        Creates elevator transitions using Django ORM with raw SQL for complex spatial operations.
        """
        elevators_use_type = "elev"
        
        sql = f"""
            WITH elevators AS (
                SELECT
                    unit_id,
                    level_id,
                    use_type,
                    geom,
                    ST_Centroid(geom) AS centroid_2d,
                    ST_Z(ST_PointN(ST_ExteriorRing(geom), 1)) AS z_val
                FROM {self.db_schema}.units
                WHERE use_type = %s
            ),
            clustered AS (
                SELECT
                    unit_id,
                    level_id,
                    use_type,
                    geom,
                    z_val,
                    ST_ClusterDBSCAN(centroid_2d, eps := 0.8, minpoints := 2) OVER () AS elevator_id,
                    ST_SetSRID(ST_MakePoint(
                        ST_X(centroid_2d),
                        ST_Y(centroid_2d),
                        ST_Z(ST_PointN(ST_ExteriorRing(geom), 1))
                    ), ST_SRID(geom)) AS elev_centroid_3d
                FROM elevators
            ),
            closest_path_points AS (
                SELECT DISTINCT ON (c.unit_id)
                    c.unit_id,
                    c.level_id,
                    c.use_type,
                    c.z_val,
                    c.elevator_id,
                    CASE 
                        WHEN ST_3DDistance(ST_StartPoint(pw.geom), c.elev_centroid_3d) <
                            ST_3DDistance(ST_EndPoint(pw.geom), c.elev_centroid_3d)
                        THEN ST_StartPoint(pw.geom)
                        ELSE ST_EndPoint(pw.geom)
                    END AS snapped_point_geom
                FROM clustered c
                JOIN {self.db_schema}.pathways pw
                ON pw.level_id = c.level_id
                AND ST_3DDWithin(pw.geom, c.elev_centroid_3d, 0.5)
                ORDER BY c.unit_id, ST_3DDistance(pw.geom, c.elev_centroid_3d)
            ),
            lines_raw AS (
                SELECT
                    elevator_id,
                    MIN(unit_id) AS unit_id,
                    MIN(use_type) AS use_type,
                    ST_MakeLine(snapped_point_geom ORDER BY z_val) AS raw_geom
                FROM closest_path_points
                GROUP BY elevator_id
            ),
            densified_interp AS (
                SELECT
                    lr.elevator_id,
                    lr.unit_id,
                    lr.use_type,
                    ST_3DLineInterpolatePoint(lr.raw_geom, p) AS pt
                FROM lines_raw lr,
                generate_series(0, 1, %s) AS p
            ),
            original_points AS (
                SELECT
                    elevator_id,
                    unit_id,
                    use_type,
                    snapped_point_geom AS pt
                FROM closest_path_points
            ),
            all_points AS (
                SELECT * FROM densified_interp
                UNION
                SELECT * FROM original_points
            ),
            final_lines AS (
                SELECT
                    elevator_id,
                    MIN(unit_id) AS unit_id,
                    MIN(use_type) AS use_type,
                    ST_MakeLine(pt ORDER BY ST_Z(pt)) AS geom
                FROM all_points
                GROUP BY elevator_id
            )
            INSERT INTO {self.db_schema}.transitions (elevator_id, unit_id, use_type, geom)
            SELECT
                elevator_id,
                unit_id,
                use_type,
                geom
            FROM final_lines;
        """
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, [elevators_use_type, self.densify_distance])
            
            self.logger.info("Elevators transitions created successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating elevator transitions: {e}")
            return False

    @transaction.atomic
    def create_stairs_transitions(self) -> bool:
        """
        Creates stairs transitions using Django ORM with raw SQL for complex spatial operations.
        """
        stairs_use_type = "stair"
        
        sql = f"""
            WITH stairs AS (
                SELECT
                    unit_id,
                    level_id,
                    use_type,
                    geom,
                    ST_Centroid(geom) AS centroid_2d,
                    ST_Z(ST_PointN(ST_ExteriorRing(geom), 1)) AS z_val
                FROM {self.db_schema}.units
                WHERE use_type = %s
            ),
            clustered AS (
                SELECT
                    unit_id,
                    level_id,
                    use_type,
                    geom,
                    z_val,
                    ST_ClusterIntersectingWin(geom) OVER () AS elevator_id,
                    ST_SetSRID(ST_MakePoint(
                        ST_X(centroid_2d),
                        ST_Y(centroid_2d),
                        ST_Z(ST_PointN(ST_ExteriorRing(geom), 1))
                    ), ST_SRID(geom)) AS elev_centroid_3d
                FROM stairs
            ),
            closest_path_points AS (
                SELECT DISTINCT ON (c.unit_id)
                    c.unit_id,
                    c.level_id,
                    c.use_type,
                    c.z_val,
                    c.elevator_id,
                    CASE 
                        WHEN ST_3DDistance(ST_StartPoint(pw.geom), c.elev_centroid_3d) <
                            ST_3DDistance(ST_EndPoint(pw.geom), c.elev_centroid_3d)
                        THEN ST_StartPoint(pw.geom)
                        ELSE ST_EndPoint(pw.geom)
                    END AS snapped_point_geom
                FROM clustered c
                JOIN {self.db_schema}.pathways pw
                ON pw.level_id = c.level_id
                AND ST_3DDWithin(pw.geom, c.elev_centroid_3d, 0.5)
                ORDER BY c.unit_id, ST_3DDistance(pw.geom, c.elev_centroid_3d)
            ),
            lines_raw AS (
                SELECT
                    elevator_id,
                    MIN(unit_id) AS unit_id,
                    MIN(use_type) AS use_type,
                    ST_MakeLine(snapped_point_geom ORDER BY z_val) AS raw_geom
                FROM closest_path_points
                GROUP BY elevator_id
            ),
            densified_interp AS (
                SELECT
                    lr.elevator_id,
                    lr.unit_id,
                    lr.use_type,
                    ST_3DLineInterpolatePoint(lr.raw_geom, p) AS pt
                FROM lines_raw lr,
                generate_series(0, 1, %s) AS p
            ),
            original_points AS (
                SELECT
                    elevator_id,
                    unit_id,
                    use_type,
                    snapped_point_geom AS pt
                FROM closest_path_points
            ),
            all_points AS (
                SELECT * FROM densified_interp
                UNION
                SELECT * FROM original_points
            ),
            final_lines AS (
                SELECT
                    elevator_id,
                    MIN(unit_id) AS unit_id,
                    MIN(use_type) AS use_type,
                    ST_MakeLine(pt ORDER BY ST_Z(pt)) AS geom
                FROM all_points
                GROUP BY elevator_id
            )
            INSERT INTO {self.db_schema}.transitions (elevator_id, unit_id, use_type, geom)
            SELECT
                elevator_id,
                unit_id,
                use_type,
                geom
            FROM final_lines;
        """
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, [stairs_use_type, self.densify_distance])
            
            self.logger.info("Stairs transitions created successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating stairs transitions: {e}")
            return False

    def fetch_levels_polygons(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetches level polygons using Django ORM.
        Returns a dictionary with level data including geometry and Z values.
        """
        levels = {}
        
        try:
            level_queryset = Level.objects.select_related().all()
            
            for level in level_queryset:
                # Extract Z value from the geometry
                z_value = None
                try:
                    if level.geom and hasattr(level.geom, 'coords'):
                        coords = list(level.geom.coords[0])  # First ring of polygon
                        if coords and len(coords[0]) == 3:
                            z_value = coords[0][2]
                except Exception as e:
                    self.logger.warning(f"Could not extract Z for level {level.level_id}: {e}")
                
                # Convert to shapely geometry for compatibility
                shapely_geom = None
                if level.geom:
                    shapely_geom = wkb.loads(level.geom.wkb)
                
                levels[str(level.level_id)] = {
                    "geometry": shapely_geom,
                    "z_value": z_value,
                    "name_short": level.name_short,
                    "django_instance": level
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching level polygons: {e}")
            
        return levels

    def fetch_barriers_polylines(self, level_id: str, barriers_list: List[str]) -> List[Any]:
        """
        Fetches barrier polylines for a specific level using Django ORM.
        """
        barriers = []
        
        try:
            details = Detail.objects.filter(
                level_id=level_id,
                use_type__in=barriers_list
            )
            
            for detail in details:
                if detail.geom:
                    # Convert to shapely geometry
                    geom = wkb.loads(detail.geom.wkb)
                    
                    # Force to 2D by dropping Z from each coordinate
                    coords_2d = [(x, y) for x, y, *rest in geom.coords]
                    line2d = ShapelyLineString(coords_2d)
                    
                    # Buffer outwards to get a polygon barrier
                    buf = line2d.buffer(self.buffer_dist)
                    barriers.append(buf)
                    
        except Exception as e:
            self.logger.error(f"Error fetching barriers for level {level_id}: {e}")
            
        return barriers

    def insert_record(self, table_name: str, geom_wkt: str, schema_name: str = None, 
                     attribute_dict: Dict[str, Any] = None) -> bool:
        """
        Inserts a record using Django ORM models.
        Maps table names to Django models and creates instances.
        """
        schema_name = schema_name or self.db_schema
        attribute_dict = attribute_dict or {}
        
        # Map table names to Django models
        model_mapping = {
            'facility': Facility,
            'levels': Level,
            'units': Unit,
            'details': Detail,
            'landmarks': Landmark,
            'pathways': Pathway,
            'transitions': Transition,
        }
        
        model_class = model_mapping.get(table_name.lower())
        if not model_class:
            self.logger.error(f"Unknown table name: {table_name}")
            return False
        
        try:
            # Create geometry from WKT
            geometry = GEOSGeometry(geom_wkt, srid=self.srid)
            
            # Create model instance
            instance_data = attribute_dict.copy()
            instance_data['geom'] = geometry
            
            instance = model_class(**instance_data)
            instance.full_clean()  # Validate the instance
            instance.save()
            
            return True
            
        except ValidationError as e:
            self.logger.error(f"Validation error inserting record into {table_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error inserting record into {table_name}: {e}")
            return False

    def tables_metadata_provider(self) -> List[Tuple[str, str, str]]:
        """
        Provides metadata about spatial tables using Django's database introspection.
        """
        try:
            with connection.cursor() as cursor:
                sql = """
                    SELECT f_table_schema, f_table_name, f_geometry_column
                    FROM geometry_columns
                    WHERE f_table_schema = %s
                """
                cursor.execute(sql, [self.db_schema])
                return cursor.fetchall()
                
        except Exception as e:
            self.logger.error(f"Error getting table metadata: {e}")
            return []

    def get_model_by_table_name(self, table_name: str):
        """
        Returns the Django model class for a given table name.
        """
        model_mapping = {
            'facility': Facility,
            'levels': Level,
            'units': Unit,
            'details': Detail,
            'landmarks': Landmark,
            'pathways': Pathway,
            'transitions': Transition,
        }
        
        return model_mapping.get(table_name.lower())

    def bulk_create_records(self, table_name: str, records: List[Dict[str, Any]]) -> bool:
        """
        Bulk creates records for better performance.
        """
        model_class = self.get_model_by_table_name(table_name)
        if not model_class:
            self.logger.error(f"Unknown table name: {table_name}")
            return False
        
        try:
            instances = []
            for record in records:
                if 'geom_wkt' in record:
                    record['geom'] = GEOSGeometry(record.pop('geom_wkt'), srid=self.srid)
                instances.append(model_class(**record))
            
            model_class.objects.bulk_create(instances, batch_size=1000)
            self.logger.info(f"Bulk created {len(instances)} records in {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error bulk creating records in {table_name}: {e}")
            return False