import uuid
from django.contrib.gis.db import models
from django.contrib.gis.geos import Point, LineString, Polygon, MultiPolygon
from django.core.validators import MinValueValidator


class Facility(models.Model):
    """Facility model with spatial geometry support."""
    elevation_relative = models.FloatField()
    facility_number = models.IntegerField(null=True, blank=True)
    levels_total = models.IntegerField(null=True, blank=True)
    facility_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.TextField(null=True, blank=True)
    site_id = models.UUIDField()
    site_name = models.TextField(null=True, blank=True)
    geom = models.MultiPolygonField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'facility'
        
    def __str__(self):
        return f"Facility {self.name or self.facility_id}"


class Level(models.Model):
    """Level model with spatial geometry support."""
    elevation_relative = models.FloatField()
    facility = models.ForeignKey(Facility, on_delete=models.CASCADE, to_field='facility_id', db_column='facility_id')
    level_id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    level_number = models.IntegerField(null=True, blank=True)
    access_type = models.TextField(null=True, blank=True)
    name = models.TextField(null=True, blank=True)
    name_short = models.TextField()
    geom = models.PolygonField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'levels'
        
    def __str__(self):
        return f"Level {self.name_short} - {self.name or self.level_id}"


class Unit(models.Model):
    """Unit model with spatial geometry support."""
    elevation_relative = models.FloatField()
    level = models.ForeignKey(Level, on_delete=models.CASCADE, to_field='level_id', db_column='level_id')
    name = models.TextField(null=True, blank=True)
    unit_id = models.TextField(primary_key=True)
    use_type = models.TextField(null=True, blank=True)
    geom = models.PolygonField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'units'
        
    def __str__(self):
        return f"Unit {self.unit_id} - {self.name or self.use_type}"


class Detail(models.Model):
    """Detail model with spatial geometry support."""
    elevation_relative = models.FloatField()
    level = models.ForeignKey(Level, on_delete=models.CASCADE, to_field='level_id', db_column='level_id')
    use_type = models.TextField(null=True, blank=True)
    detail_id = models.TextField(primary_key=True)
    ocls_extruded_base = models.IntegerField(null=True, blank=True)
    olcs_extruded_height = models.IntegerField(null=True, blank=True)
    geom = models.LineStringField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'details'
        
    def __str__(self):
        return f"Detail {self.detail_id} - {self.use_type}"


class Landmark(models.Model):
    """Landmark model with spatial geometry support."""
    elevation_relative = models.FloatField()
    level = models.ForeignKey(Level, on_delete=models.CASCADE, to_field='level_id', db_column='level_id')
    name = models.TextField(null=True, blank=True)
    unit_id = models.TextField()
    use_type = models.TextField(null=True, blank=True)
    geom = models.PointField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'landmarks'
        
    def __str__(self):
        return f"Landmark {self.name or self.unit_id}"


class Pathway(models.Model):
    """Pathway model with spatial geometry support."""
    id = models.AutoField(primary_key=True)
    level = models.ForeignKey(Level, on_delete=models.CASCADE, to_field='level_id', db_column='level_id')
    geom = models.LineStringField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'pathways'
        
    def __str__(self):
        return f"Pathway {self.id} on Level {self.level.name_short}"


class Transition(models.Model):
    """Transition model with spatial geometry support."""
    elevator_id = models.IntegerField(null=True, blank=True)
    unit_id = models.TextField(null=True, blank=True)
    use_type = models.TextField(null=True, blank=True)
    geom = models.LineStringField(dim=3, srid=None)  # Will be set based on settings

    class Meta:
        db_table = 'transitions'
        
    def __str__(self):
        return f"Transition {self.elevator_id} - {self.use_type}"