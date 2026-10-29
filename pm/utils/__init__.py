"""Leaf utilities shared across runtime package boundaries.

``operating_system.OperatingSystem`` owns native process, signal, lock, and
low-level file operations. ``yaml_util`` owns safe YAML encoding and decoding.
These modules import no other PM package and contain no companion behavior.
"""
