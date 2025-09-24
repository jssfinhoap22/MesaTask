#!/usr/bin/env python3
"""
Script for reconstructing 3D scenes from layout JSON using Blender.
"""

import bpy
import json
import os
import time
import argparse
import sys
import math
import numpy as np
from mathutils import Quaternion, Vector, Matrix

# Global constants
PLACEMENT_HEIGHT = 0.01  # Height for placement area
RESTRICTED_HEIGHT = 0.005  # Height for restricted area

def clear_scene():
    """Clear all objects in current Blender scene except cameras and lights"""
    bpy.ops.object.select_all(action='SELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in {'CAMERA', 'LIGHT'}:
            obj.select_set(False)
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()

def get_object_dimensions(obj):
    """Calculate exact bounding box dimensions in local coordinates"""
    if obj.type != 'MESH' or not obj.data.vertices:
        return Vector((0, 0, 0))

    mesh = obj.data
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for vert in mesh.vertices:
        v_co = vert.co
        min_coords.x = min(min_coords.x, v_co.x)
        min_coords.y = min(min_coords.y, v_co.y)
        min_coords.z = min(min_coords.z, v_co.z)
        max_coords.x = max(max_coords.x, v_co.x)
        max_coords.y = max(max_coords.y, v_co.y)
        max_coords.z = max(max_coords.z, v_co.z)

    if float('inf') in min_coords or float('-inf') in max_coords:
        return Vector((0,0,0))  # Empty mesh

    return max_coords - min_coords

def process_one_instance_bpy(i, uid, position_zup, rotation_quat_zup, target_size_zup, model_base_path, instance_id):
    """
    Process a single object instance in Blender.

    Args:
        i (int): Object index
        uid (str): Model UID to load
        position_zup (list): Target position in Z-up coordinates [x, y, z]
        rotation_quat_zup (list): Target rotation quaternion in Z-up coordinates [x, y, z, w]
        target_size_zup (list): Target size in Z-up coordinates [width, depth, height]
        model_base_path (str): Base path for model files
        instance_id (str): Original instance ID for naming

    Returns:
        bpy.types.Object or None: Processed Blender object, None if failed
    """
    glb_path = os.path.join(model_base_path, f"{uid}.glb")

    if not os.path.exists(glb_path):
        print(f"  File not found: {glb_path}")
        return None


    t0 = time.time()

    # --- 0. Import GLB ---
    # Record objects before import
    objects_before = set(bpy.context.scene.objects)
    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=glb_path)
    # Identify new imported objects
    objects_after = set(bpy.context.scene.objects)
    imported_objects = list(objects_after - objects_before)

    # Filter for mesh objects
    imported_meshes = [obj for obj in imported_objects if obj.type == 'MESH']

    if not imported_meshes:
        print(f"  Warning: No mesh objects found in {glb_path}. Skipping instance {instance_id} ({uid}).")
        # Clean up potentially imported non-mesh objects (lights, cameras, etc.)
        for obj in imported_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        return None

    # --- 1. Set origin to geometric center ---
    # Deselect all objects, then select all imported meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_meshes:
        obj.select_set(True)
    # Set active object (merge operation requires)
    bpy.context.view_layer.objects.active = imported_meshes[0]

    # Apply object's current transform to calculate bounding box based on geometry itself
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Calculate combined bounding box for all selected meshes
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in imported_meshes:
        # Use object's world coordinate bounding box corners (after transform, its local coordinates are world coordinates)
        for corner_local in obj.bound_box:
            corner_world = obj.matrix_world @ Vector(corner_local)
            min_coord.x = min(min_coord.x, corner_world.x)
            min_coord.y = min(min_coord.y, corner_world.y)
            min_coord.z = min(min_coord.z, corner_world.z)
            max_coord.x = max(max_coord.x, corner_world.x)
            max_coord.y = max(max_coord.y, corner_world.y)
            max_coord.z = max(max_coord.z, corner_world.z)

    # Check if bounding box is valid
    if float('inf') in min_coord or float('-inf') in max_coord:
        print(f"  Warning: Could not calculate a valid combined bounding box for object {uid} ({instance_id}).")
        bpy.ops.object.delete() # Delete the selected (i.e., imported) mesh
        return None

    # Calculate center of combined bounding box
    bbox_center = (min_coord + max_coord) / 2.0

    # Move 3D cursor to the calculated center point
    bpy.context.scene.cursor.location = bbox_center

    # Set origin of all selected mesh objects to the 3D cursor position
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN') # Use 'ORIGIN_CURSOR'

    # Reset 3D cursor position
    bpy.context.scene.cursor.location = (0, 0, 0)

    # Merge all selected mesh objects into one
    # Ensure the active object is one of the objects to be merged
    if bpy.context.view_layer.objects.active not in imported_meshes:
            bpy.context.view_layer.objects.active = imported_meshes[0]
    bpy.ops.object.join()

    # The merged object is now the active object
    mesh_obj = bpy.context.object

    # --- 3. Calculate and apply scaling ---
    # Get initial size of the merged object (origin is already at geometric center)
    initial_dimensions = get_object_dimensions(mesh_obj)

    # Check if initial dimensions are valid
    if initial_dimensions.length < 1e-6 or any(d < 1e-6 for d in initial_dimensions):
        print(f"  Warning: Initial size of object {uid} ({instance_id}) after merging is too small or invalid: {initial_dimensions.to_tuple(3)}. Skipping scaling.")
        scale_factors = Vector((1.0, 1.0, 1.0)) # Do not scale
    else:
        target_width, target_depth, target_height = target_size_zup
        target_dimensions = Vector((target_width, target_depth, target_height))

        print(f"  Target size: {target_dimensions.to_tuple(3)}, Initial size: {initial_dimensions.to_tuple(3)}")

        scale_factors = Vector((1.0, 1.0, 1.0))
        valid_scale = True
        for axis in range(3):
            if initial_dimensions[axis] > 1e-6: # Avoid division by zero or very small numbers
                scale_factors[axis] = target_dimensions[axis] / initial_dimensions[axis]
            elif target_dimensions[axis] > 1e-6: # Initial is zero but target is not, cannot scale
                print(f"  Warning: Object {uid} ({instance_id}) initial size is near zero on axis {axis}, but target size is not zero. Cannot calculate valid scaling.")
                valid_scale = False
                break
            # If both initial and target sizes are near zero, keep scale factor as 1.0

        if not valid_scale:
            scale_factors = Vector((1.0, 1.0, 1.0)) # If valid scaling cannot be calculated, reset to no scaling

    # Apply scaling (now applied to the single merged object)
    mesh_obj.scale = scale_factors

    # --- 2. Apply rotation ---
    if rotation_quat_zup:
        # Blender uses (w, x, y, z) order
        x, y, z, w = rotation_quat_zup
        # import pdb; pdb.set_trace()
        quat = Quaternion((w, x, y, z))

        # Check and normalize quaternion
        if abs(quat.magnitude - 1.0) > 1e-3:
                print(f"  Warning: Quaternion norm {quat.magnitude:.4f} of object {uid} ({instance_id}) is not close to 1. Normalizing.")
                if quat.magnitude > 1e-6:
                    quat.normalize()
                else:
                    print(f"  Error: Quaternion norm is too small, cannot normalize. Skipping rotation.")
                    quat = Quaternion((1, 0, 0, 0)) # No rotation

        mesh_obj.rotation_mode = 'QUATERNION'
        mesh_obj.rotation_quaternion = quat
    # (If z_rotation angle is needed, you can add an else branch here to handle it)
    # else:
    #    z_angle_rad = math.radians(z_rotation_degrees)
    #    mesh_obj.rotation_mode = 'XYZ' # Or 'ZXY', etc., depending on your needs
    #    mesh_obj.rotation_euler = (0, 0, z_angle_rad)






    # --- 4. Apply translation ---
    # Since origin is already at geometric center, just set object location
    mesh_obj.location = Vector(position_zup)

    # --- 5. Name object ---
    # Clean up potentially invalid characters
    safe_instance_id = "".join(c if c.isalnum() else "_" for c in instance_id)
    safe_uid = "".join(c if c.isalnum() else "_" for c in uid)
    base_name = f"{safe_instance_id}_{safe_uid}"
    mesh_obj.name = base_name
    # Blender will automatically handle duplicates, adding .001, .002, etc.

    # print(f"  Object {i} ({uid}, {instance_id}) processed. Target size: {target_size_zup}, Initial size: {initial_dimensions.to_tuple(3)}, Calculated scale: {scale_factors.to_tuple(3)}")
    return mesh_obj
    

def create_item_placement_zone_bpy(bbox, table_path=None):
    """Create table model representing item_placement_zone in Blender"""
    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    depth = y_max - y_min # Z-up depth is along Y axis
    height = PLACEMENT_HEIGHT

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    # In Blender Z-up, object center is at its geometric center
    # We want the object top to be at Z=0, so center is at Z = -height / 2
    center_z = -height / 2

    if table_path is None:
        print(f"  Warning: No table model path provided. Skipping creation.")
        return None

    if not os.path.exists(table_path):
        print(f"  Warning: Table model not found: {table_path}. Skipping creation.")
        return None

    try:
        objects_before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=table_path)
        objects_after = set(bpy.context.scene.objects)
        imported_objects = list(objects_after - objects_before)

        # Filter for mesh objects
        imported_meshes = [obj for obj in imported_objects if obj.type == 'MESH']
        
        if not imported_meshes:
            print(f"  Warning: No mesh objects found in {table_path}. Cannot create item_placement_zone.")
            for obj in imported_objects: bpy.data.objects.remove(obj, do_unlink=True)
            return None

        # Deselect all objects, then select all imported meshes
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_meshes:
            obj.select_set(True)
        
        # Set active object (merge operation requires)
        bpy.context.view_layer.objects.active = imported_meshes[0]
        
        # Apply transforms to all objects
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Merge all mesh objects into one
        if len(imported_meshes) > 1:
            bpy.ops.object.join()
        
        # The merged object is the active object
        table_mesh_obj = bpy.context.active_object
        
        # Set origin to geometric center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        table_mesh_obj.location = (0, 0, 0)
        table_mesh_obj.rotation_euler = (0, 0, 0) # Clear any imported rotation

        # Calculate scaling
        table_mesh_obj.scale = (1, 1, 1)
        initial_dimensions = get_object_dimensions(table_mesh_obj)
        target_dimensions = Vector((width, depth, height))
        
        scale_factors = Vector((1.0, 1.0, 1.0))
        valid_scale = True
        for axis in range(3):
            if initial_dimensions[axis] > 1e-6:
                scale_factors[axis] = target_dimensions[axis] / initial_dimensions[axis]
            elif target_dimensions[axis] > 1e-6:
                valid_scale = False; break
        
        if not valid_scale:
             print(f"  Warning: Could not calculate valid scaling for table model {table_path}. Skipping item_placement_zone.")
             bpy.data.objects.remove(table_mesh_obj, do_unlink=True)
             return None

        table_mesh_obj.scale = scale_factors
        
        # Set position
        table_mesh_obj.location = (center_x, center_y, center_z)
        table_mesh_obj.name = "table"


        # 2. Create new material
        mat_name = "PlacementZoneDarkGreyMat"
        dark_grey_material = bpy.data.materials.new(name=mat_name)
        dark_grey_material.use_nodes = True # Ensure nodes are used


        print(f"  item_placement_zone created.")
        return table_mesh_obj

    except Exception as e:
        import traceback
        print(f"  Error creating item_placement_zone: {e}")
        print(traceback.format_exc())
        if 'imported_objects' in locals():
             for obj in imported_objects:
                  if obj.name in bpy.data.objects:
                      bpy.data.objects.remove(obj, do_unlink=True)
        return None

def create_restricted_zone_bpy(bbox, sink_path=None):
    """Create object representing restricted_zone in Blender"""
    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    depth = y_max - y_min
    height = RESTRICTED_HEIGHT

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
     # We want the object bottom to be at Z=0, so center is at Z = height / 2
    center_z = - height / 2

    if sink_path is None:
        print(f"  Warning: No restricted area model path provided. Skipping creation.")
        return None

    if not os.path.exists(sink_path):
        print(f"  Warning: Restricted area model not found: {sink_path}. Skipping creation.")
        return None

    try:
        objects_before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=sink_path)
        objects_after = set(bpy.context.scene.objects)
        imported_objects = list(objects_after - objects_before)

        sink_mesh_obj = None
        for obj in imported_objects:
            if obj.type == 'MESH':
                sink_mesh_obj = obj
                break
        
        if not sink_mesh_obj:
            print(f"  Warning: No mesh object found in {sink_path}. Cannot create restricted_zone.")
            for obj in imported_objects: bpy.data.objects.remove(obj, do_unlink=True)
            return None

        bpy.context.view_layer.objects.active = sink_mesh_obj
        bpy.ops.object.select_all(action='DESELECT')
        sink_mesh_obj.select_set(True)
        
        # Set origin and move to world origin
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        sink_mesh_obj.location = (0, 0, 0)
        sink_mesh_obj.rotation_euler = (0, 0, 0) # Clear any imported rotation

        # Calculate scaling
        sink_mesh_obj.scale = (1, 1, 1)
        initial_dimensions = get_object_dimensions(sink_mesh_obj)
        target_dimensions = Vector((width, depth, height))
        
        scale_factors = Vector((1.0, 1.0, 1.0))
        valid_scale = True
        for axis in range(3):
            if initial_dimensions[axis] > 1e-6:
                scale_factors[axis] = target_dimensions[axis] / initial_dimensions[axis]
            elif target_dimensions[axis] > 1e-6:
                valid_scale = False; break
        
        if not valid_scale:
             print(f"  Warning: Could not calculate valid scaling for sink model {sink_path}. Skipping restricted_zone.")
             bpy.data.objects.remove(sink_mesh_obj, do_unlink=True)
             return None

        sink_mesh_obj.scale = scale_factors
        
        # Set position
        sink_mesh_obj.location = (center_x, center_y, center_z)
        sink_mesh_obj.name = "restricted_zone"

        # --- Use user-provided similar method to set grey material (English API) ---
        # 1. Clear all existing material slots on the object (ensure clean state)
        sink_mesh_obj.data.materials.clear()

        # 2. Create a new material
        # bpy.ops.material.new() can also be used, but bpy.data is more direct
        # We give it a name so it can be found later, and Blender will automatically add .001, etc. if it's a duplicate
        new_mat_name = "RestrictedZoneGreyMat"
        grey_material = bpy.data.materials.new(name=new_mat_name)
        grey_material.use_nodes = True # Ensure nodes are used

        # 3. Assign the new material to the object
        sink_mesh_obj.data.materials.append(grey_material)

        # 4. Get the material and its node tree
        mat = bpy.data.materials.get(new_mat_name)
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            # --- Temporarily replace with Emission ---
            # Clear old nodes
            for node in nodes:
                nodes.remove(node)
            # Add Emission and Output nodes
            emission = nodes.new(type='ShaderNodeEmission')
            emission.inputs['Color'].default_value = (0.1, 0.1, 0.8, 1.0) # Set emission to grey
            emission.inputs['Strength'].default_value = 1.0
            emission.location = (0, 0)
            material_output = nodes.new(type='ShaderNodeOutputMaterial')
            material_output.location = (250, 0)
            # Connect nodes
            links.new(emission.outputs['Emission'], material_output.inputs['Surface'])
            print(f"  Applied temporary Emission material '{new_mat_name}'")
            # --- Emission replacement ends ---

        # Comment out the original Principled BSDF setting code
        # if mat and mat.use_nodes:
        #     nodes = mat.node_tree.nodes
        #     principled_bsdf = nodes.get("Principled BSDF")
        #     # ... (original code) ...
        # else:
        #      print(f"  Error: Material '{new_mat_name}' does not use nodes.")

        # --- Check vertex colors (keep previous check) ---
        if sink_mesh_obj.data.vertex_colors:
             active_vc_layer = sink_mesh_obj.data.vertex_colors.active
             if active_vc_layer:
                 print(f"  Warning: Object '{sink_mesh_obj.name}' has active vertex color layer ('{active_vc_layer.name}'). This might override material color in some views.")
        # --- Material setting ends ---

        return sink_mesh_obj

    except Exception as e:
        import traceback
        print(f"  Error creating restricted_zone: {e}")
        print(traceback.format_exc())
        if 'imported_objects' in locals():
             for obj in imported_objects:
                  if obj.name in bpy.data.objects:
                      bpy.data.objects.remove(obj, do_unlink=True)
        return None

def get_scene_bounds(ignore_types={'CAMERA', 'LIGHT'}):
    """
    Calculate combined world-space bounding box for the scene with specific rules:
    - XY bounds determined by 'item_placement_zone'
    - Z min bound set to max Z of 'item_placement_zone'
    - Z max bound determined by highest point of other objects
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Initialize variables
    placement_zone_bounds = {
        "min_x": float('inf'), "min_y": float('inf'),
        "max_x": float('-inf'), "max_y": float('-inf'),
        "max_z": float('-inf'), # We need max Z of placement zone as scene's min Z
        "min_z_actual": float('inf') # Also record its actual min Z, just in case
    }
    max_z_others = float('-inf') # Max Z of other objects
    placement_zone_found = False
    other_objects_found = False

    # Iterate through original objects in the scene
    for obj_orig in bpy.context.scene.objects:
        if obj_orig.type in ignore_types or obj_orig.hide_render:
            continue
        if obj_orig.type == 'MESH' and not obj_orig.data.vertices:
            continue

        obj_eval = obj_orig.evaluated_get(depsgraph)

        try:
            world_bbox_corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
        except Exception as e:
            print(f"  Warning: Could not get bounding box for object {obj_orig.name}: {e}. Skipping this object.")
            continue

        if not world_bbox_corners or len(world_bbox_corners) != 8:
            print(f"  Warning: Bounding box corners for object {obj_orig.name} are invalid or incomplete. Skipping this object.")
            continue

        # Calculate local world coordinate bounding box for current object
        obj_min = Vector((float('inf'), float('inf'), float('inf')))
        obj_max = Vector((float('-inf'), float('-inf'), float('-inf')))
        for corner in world_bbox_corners:
            obj_min.x = min(obj_min.x, corner.x)
            obj_min.y = min(obj_min.y, corner.y)
            obj_min.z = min(obj_min.z, corner.z)
            obj_max.x = max(obj_max.x, corner.x)
            obj_max.y = max(obj_max.y, corner.y)
            obj_max.z = max(obj_max.z, corner.z)

        # Check if it's item_placement_zone
        # Use startswith to avoid Blender automatically adding .001, etc. suffixes
        if obj_orig.name.startswith("item_placement_zone"):
            placement_zone_found = True
            # Update placement zone bounds
            placement_zone_bounds["min_x"] = min(placement_zone_bounds["min_x"], obj_min.x)
            placement_zone_bounds["min_y"] = min(placement_zone_bounds["min_y"], obj_min.y)
            placement_zone_bounds["max_x"] = max(placement_zone_bounds["max_x"], obj_max.x)
            placement_zone_bounds["max_y"] = max(placement_zone_bounds["max_y"], obj_max.y)
            placement_zone_bounds["max_z"] = max(placement_zone_bounds["max_z"], obj_max.z)
            placement_zone_bounds["min_z_actual"] = min(placement_zone_bounds["min_z_actual"], obj_min.z)
            print(f"  Found item_placement_zone: {obj_orig.name}, XY=({obj_min.x:.2f},{obj_min.y:.2f})-({obj_max.x:.2f},{obj_max.y:.2f}), Z=({obj_min.z:.2f}-{obj_max.z:.2f})")
        elif not obj_orig.name.startswith("restricted_zone"): # Exclude restricted_zone
             other_objects_found = True
             # Update max Z value for other objects
             max_z_others = max(max_z_others, obj_max.z)
             print(f"  Considering other objects: {obj_orig.name}, Z_max={obj_max.z:.2f}")


    # --- Finalize bounding box ---
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    if placement_zone_found:
        print("  Using item_placement_zone to define XY bounds and Z min.")
        min_coord.x = placement_zone_bounds["min_x"]
        min_coord.y = placement_zone_bounds["min_y"]
        # Set scene's min Z to placement zone's max Z as per user's request
        min_coord.z = placement_zone_bounds["max_z"]
        # min_coord.z = placement_zone_bounds["min_z_actual"] # Alternative: use placement zone's actual lowest point

        max_coord.x = placement_zone_bounds["max_x"]
        max_coord.y = placement_zone_bounds["max_y"]

        if other_objects_found:
            # Z max value determined by other objects
            # Also ensure Z max is at least not lower than Z min (placement zone's max_z)
            max_coord.z = max(max_z_others, min_coord.z)
            print(f"  Highest Z of other objects: {max_z_others:.2f}.")
        else:
            # If no other objects, then Z max is also determined by placement zone (or set to same as Z min)
            max_coord.z = min_coord.z # Or placement_zone_bounds["max_z"]
            print("  No other objects found, Z max set to Z min.")

    else:
        # --- Fallback logic: if item_placement_zone not found ---
        print("Warning: 'item_placement_zone' object not found. Using combined bounding box of all objects.")
        # Re-iterate to calculate combined bounding box of all objects
        has_objects = False
        for obj_orig in bpy.context.scene.objects:
            # (Repeat previous iteration and bounding box calculation logic, but do not distinguish placement_zone)
            if obj_orig.type in ignore_types or obj_orig.hide_render: continue
            if obj_orig.type == 'MESH' and not obj_orig.data.vertices: continue
            obj_eval = obj_orig.evaluated_get(depsgraph)
            try: world_bbox_corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
            except Exception: continue
            if not world_bbox_corners or len(world_bbox_corners) != 8: continue

            has_objects = True
            for corner in world_bbox_corners:
                min_coord.x = min(min_coord.x, corner.x)
                min_coord.y = min(min_coord.y, corner.y)
                min_coord.z = min(min_coord.z, corner.z)
                max_coord.x = max(max_coord.x, corner.x)
                max_coord.y = max(max_coord.y, corner.y)
                max_coord.z = max(max_coord.z, corner.z)
        if not has_objects: # If fallback logic also found no objects
             print("Warning: Fallback logic also found no valid objects.")
             min_coord = Vector((0,0,0)); max_coord = Vector((0,0,0))


    # Check if final bounds are valid
    if float('inf') in min_coord or float('-inf') in max_coord:
        print("Warning: Could not calculate a valid scene bounding box.")
        return None, None, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))

    center = (min_coord + max_coord) / 2.0
    size = max_coord - min_coord
    # Ensure size is not negative (can happen if Zmin > Zmax)
    size.x = max(size.x, 0)
    size.y = max(size.y, 0)
    size.z = max(size.z, 0)

    print(f"  Final calculated bounds: Min={min_coord.to_tuple(3)}, Max={max_coord.to_tuple(3)}")
    print(f"  Final calculated center: {center.to_tuple(3)}")
    print(f"  Final calculated size: {size.to_tuple(3)}")
    return min_coord, max_coord, center, size

def setup_compositor_black_background():
    """Configure compositor nodes to convert transparent background to black in render output"""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear existing nodes in case
    for node in nodes:
        nodes.remove(node)

    # Create Render Layers node
    render_layers = nodes.new('CompositorNodeRLayers')
    render_layers.location = (-300, 300)

    # Create Alpha Over node
    alpha_over = nodes.new('CompositorNodeAlphaOver')
    alpha_over.location = (0, 300)
    alpha_over.inputs[1].default_value = (0, 0, 0, 1) # Background color is black

    # Connect nodes
    links.new(render_layers.outputs["Image"], alpha_over.inputs[2]) # Render result as foreground

    # Create Composite node
    composite = nodes.new('CompositorNodeComposite')
    composite.location = (300, 300)
    links.new(alpha_over.outputs["Image"], composite.inputs["Image"]) # Connect to final output

def save_camera_parameters(cam, view_name, output_dir, resolution):
    """Save camera intrinsic and extrinsic parameters to JSON file"""
    # Get camera data
    cam_data = cam.data
    scene = bpy.context.scene
    
    # Calculate intrinsic matrix
    # Blender's focal length is in millimeters, needs to be converted to pixels
    focal_length_mm = cam_data.lens
    sensor_width_mm = cam_data.sensor_width
    
    # Calculate pixel focal length
    focal_length_px = (focal_length_mm * resolution) / sensor_width_mm
    
    # Camera intrinsic matrix K
    cx = resolution / 2.0  # Principal point x-coordinate (image center)
    cy = resolution / 2.0  # Principal point y-coordinate (image center)
    
    intrinsic_matrix = [
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ]
    
    # Get camera's world transformation matrix
    world_matrix = cam.matrix_world.copy()
    
    # Camera extrinsic - world to camera transformation matrix
    # Blender uses right-handed coordinate system, camera looks towards -Z direction
    extrinsic_matrix = world_matrix.inverted()
    
    # Convert to list of 4x4 matrices
    extrinsic_matrix_list = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(extrinsic_matrix[i][j])
        extrinsic_matrix_list.append(row)
    
    # Camera position (world coordinates)
    camera_location = [cam.location.x, cam.location.y, cam.location.z]
    
    # Camera rotation (quaternion)
    camera_rotation_quat = [
        cam.rotation_quaternion.w,
        cam.rotation_quaternion.x, 
        cam.rotation_quaternion.y,
        cam.rotation_quaternion.z
    ]
    
    # Camera rotation (Euler angles, degrees)
    camera_rotation_euler = [
        math.degrees(cam.rotation_euler.x),
        math.degrees(cam.rotation_euler.y),
        math.degrees(cam.rotation_euler.z)
    ]
    
    # Assemble camera parameters
    camera_params = {
        "view_name": view_name,
        "resolution": {
            "width": resolution,
            "height": resolution
        },
        "intrinsic": {
            "focal_length_mm": focal_length_mm,
            "focal_length_px": focal_length_px,
            "sensor_width_mm": sensor_width_mm,
            "principal_point": [cx, cy],
            "matrix": intrinsic_matrix
        },
        "extrinsic": {
            "matrix": extrinsic_matrix_list,
            "camera_location": camera_location,
            "camera_rotation_quaternion": camera_rotation_quat,
            "camera_rotation_euler_degrees": camera_rotation_euler
        },
        "coordinate_system": "blender_zup_right_handed",
        "camera_direction_info": "camera looks along negative Z axis in local coordinates"
    }
    
    # Save to JSON file
    camera_params_path = os.path.join(output_dir, f"{view_name}_camera.json")
    with open(camera_params_path, 'w', encoding='utf-8') as f:
        json.dump(camera_params, f, indent=2, ensure_ascii=False)
    
    print(f"  Camera parameters saved: {camera_params_path}")

def render_four_views(output_dir, resolution=512, base_distance_factor=1.5, min_distance=0.2):
    """
    Render four standard views of current scene using Cycles GPU renderer.
    Automatically aligns to scene center and adjusts distance.
    """
    print(f"\nStarting to render four views to: {output_dir}")
    if not output_dir:
        print("Error: No valid output directory provided.")
        return

    context = bpy.context
    scene = context.scene
    render = scene.render

    # --- 1. Configure renderer (similar to previous) ---
    render.engine = 'CYCLES'
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.image_settings.file_format = 'PNG'
    render.image_settings.color_mode = 'RGBA'
    render.film_transparent = True

    # --- Add basic world background light ---
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    try: # Add try-except to prevent errors if node tree is empty
        tree = world.node_tree
        bg_node = tree.nodes.get('Background')
        if not bg_node: # If no background node, try to create
            # Clear potentially existing old nodes
            for node in tree.nodes:
                tree.nodes.remove(node)
            # Create necessary nodes
            bg_node = tree.nodes.new(type='ShaderNodeBackground')
            output_node = tree.nodes.new(type='ShaderNodeOutputWorld')
            tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

        if bg_node:
            bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0) # Set weak grey
            bg_node.inputs['Strength'].default_value = 10.0
            print("  Basic world background light set.")
        else:
             print("  Warning: Could not find or create world background node.")
    except Exception as e_world:
        print(f"  Error setting world background: {e_world}")
    # --- World background light setting ends ---

    # GPU settings (similar to previous)
    scene.cycles.device = 'GPU'
    prefs = context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA' # Or other
    enabled_gpu = False
    try: # Avoid errors if 'cycles' plugin is not installed
        for device_type in prefs.get_device_types(context):
             prefs.get_devices_for_type(device_type[0])
        for device in prefs.devices:
            if device.type == 'CUDA': # Or other
                device.use = True
                enabled_gpu = True
                print(f"  Enabled GPU device: {device.name}")
            else:
                device.use = False
    except Exception as e:
         print(f"  Error getting GPU devices: {e}, using CPU.")
         enabled_gpu = False

    if not enabled_gpu:
        print("Warning: No compatible GPU device found or enabled, using CPU for rendering.")
        scene.cycles.device = 'CPU'

    # Cycles settings (similar to previous)
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.transmission_bounces = 8
    scene.render.use_persistent_data = True

    # Configure compositor
    setup_compositor_black_background()

    # --- 2. Calculate scene bounds and center ---
    _, _, scene_center, scene_size = get_scene_bounds()
    print(f"  Calculated scene center: {scene_center.to_tuple(3)}")
    print(f"  Calculated scene size: {scene_size.to_tuple(3)}")

    # Dynamically calculate camera distance
    max_scene_dimension = max(scene_size) if any(d > 1e-6 for d in scene_size) else 1.0
    dynamic_distance = max(max_scene_dimension * base_distance_factor, min_distance)
    print(f"  Dynamically adjusted camera distance: {dynamic_distance:.2f}")


    # --- 3. Set lights (relative to scene center) ---
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    light_data = bpy.data.lights.new(name="AreaLight", type='AREA')
    light_data.energy = 50 # Try increasing this energy value, e.g., to 5000 or higher
    light_data.size = dynamic_distance # Make light size relative to distance
    light_object = bpy.data.objects.new(name="AreaLight", object_data=light_data)
    scene.collection.objects.link(light_object)
    # Place light at a slight angle above scene center
    light_object.location = scene_center + Vector((dynamic_distance * 0.8, -dynamic_distance * 0.8, dynamic_distance * 1.2))
    # Make light face towards scene center
    look_at_direction = scene_center - light_object.location
    light_object.rotation_euler = look_at_direction.to_track_quat('-Z', 'Y').to_euler()


    # --- 4. Set camera ---
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # --- 5. Define relative view direction vectors and render ---
    # These are relative view direction vectors (from this direction towards the center)
    # (x, y, z)
    view_vectors = {
        "front": Vector((0, -1, 0.6)).normalized(),      # Look slightly above front
        "left": Vector((-1, 0, 0.3)).normalized(),       # Look slightly above left
        "top": Vector((0, 0, 1)).normalized(),           # Look from top
        "perspective": Vector((-0.7, -0.7, 0.7)).normalized() # Look from a diagonal direction
    }

    os.makedirs(output_dir, exist_ok=True)

    for view_name, direction_vec in view_vectors.items():
        print(f"  Rendering view: {view_name}...")
        # Calculate camera position: scene center + direction vector * distance
        cam.location = scene_center + direction_vec * dynamic_distance

        # Calculate camera look-at direction: from camera position towards scene center
        look_at_direction = scene_center - cam.location
        cam.rotation_euler = look_at_direction.to_track_quat('-Z', 'Y').to_euler()

        # Force camera rotation mode to quaternion to get accurate parameters
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = look_at_direction.to_track_quat('-Z', 'Y')

        # Force update of scene and transformation matrices
        bpy.context.view_layer.update()
        # Or use a more complete update
        # bpy.context.evaluated_depsgraph_get().update()

        # Save camera parameters
        save_camera_parameters(cam, view_name, output_dir, resolution)

        render.filepath = os.path.join(output_dir, f"{view_name}.png")
        try:
            bpy.ops.render.render(write_still=True)
            print(f"    Image saved: {render.filepath}")
        except Exception as e:
            print(f"  Error rendering view {view_name}: {e}")
            # You can choose to continue rendering other views or stop here
            continue

    # Clean up (optional)
    if cam.name in scene.collection.objects: scene.collection.objects.unlink(cam)
    if cam.name in bpy.data.objects: bpy.data.objects.remove(cam)
    if cam_data.name in bpy.data.cameras: bpy.data.cameras.remove(cam_data)
    if light_object.name in scene.collection.objects: scene.collection.objects.unlink(light_object)
    if light_object.name in bpy.data.objects: bpy.data.objects.remove(light_object)
    if light_data.name in bpy.data.lights: bpy.data.lights.remove(light_data)

    print("Four view rendering and camera parameter saving completed.")

def reconstruct_scene_from_json_bpy(scene_json_path, model_base_path, output_glb_path=None, 
                                  render_output_dir=None, save_glb=False, render_views=False, 
                                  table_path=None, sink_path=None):
    """Reconstruct scene from JSON using Blender"""
    # Clear scene
    clear_scene()

    # Read JSON file
    if not os.path.exists(scene_json_path):
        print(f"Error: Scene JSON file not found: {scene_json_path}")
        return False # Return success status

    try:
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
        # --- REMOVED results_data loading ---
    except json.JSONDecodeError as e:
        print(f"Error: Error parsing scene JSON file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error reading scene JSON file: {e}")
        return False

    # Create scene collection (optional, for organization)
    scene_collection = bpy.context.scene.collection
    # new_collection = bpy.data.collections.new("ReconstructedScene")
    # scene_collection.children.link(new_collection)
    # layer_collection = bpy.context.view_layer.layer_collection.children[new_collection.name]
    # bpy.context.view_layer.active_layer_collection = layer_collection


    # Process item_placement_zone
    if "item_placement_zone" in scene_data:
        print("Creating item_placement_zone...")
        placement_zone_obj = create_item_placement_zone_bpy(scene_data["item_placement_zone"], table_path)
        if placement_zone_obj:
            print("  item_placement_zone created.")
            # link object to collection if using new_collection
            # new_collection.objects.link(placement_zone_obj)
            # scene_collection.objects.unlink(placement_zone_obj) # Unlink from default

    # Process restricted_zone
    if "restricted_zone" in scene_data:
        print("Creating restricted_zone...")
        restricted_zone_obj = create_restricted_zone_bpy(scene_data["restricted_zone"], sink_path)
        if restricted_zone_obj:
            print(f"  restricted_zone created (using {os.path.basename(sink_path) if sink_path else 'default'}).")
            # link object to collection if using new_collection
            # new_collection.objects.link(restricted_zone_obj)
            # scene_collection.objects.unlink(restricted_zone_obj)

    # Process all objects
    skipped_instances_log = []
    processed_count = 0
    objects_to_process = scene_data.get("objects", [])
    total_objects = len(objects_to_process)

    print(f"\nStarting to process {total_objects} objects...")

    for i, obj_data in enumerate(objects_to_process):
        instance_id = obj_data.get("instance")
        position_zup = obj_data.get("position", [0, 0, 0])
        scale_factor = obj_data.get("scale_factor") # [width, depth, height] in Z-up
        target_size_zup = obj_data.get("size") # [width, depth, height] in Z-up

        print(f"\nProcessing object: {instance_id} [{i+1}/{total_objects}]")

        # --- MODIFIED: Get retrieved_uid directly from obj_data ---
        selected_uid = obj_data.get("retrieved_uid")
        if selected_uid:
            print(f"  Using retrieved_uid from scene JSON: {selected_uid}")
        else:
            print(f"  Warning: 'retrieved_uid' not found in object '{instance_id}' in scene JSON. Skipping this object.")
            skipped_instances_log.append(f"{instance_id} (no retrieved_uid)")
            continue
        # --- END MODIFICATION ---

        # Get rotation information (logic remains the same)
        rotation_quat_zup = obj_data.get("rotation")
        # if rotation_quat_zup is None:
        #     z_angle_deg = obj_data.get("z_rotation", 0.0)
        #     z_angle_rad = math.radians(z_angle_deg)
        #     rotation_quat_zup = Quaternion((math.cos(z_angle_rad / 2.0), 0, 0, math.sin(z_angle_rad / 2.0)))
        #     rotation_quat_zup = [rotation_quat_zup[1], rotation_quat_zup[2], rotation_quat_zup[3], rotation_quat_zup[0]] # xyzw
        #     print(f"  Using z_rotation {z_angle_deg}° to generate quaternion (xyzw): {[f'{x:.4f}' for x in rotation_quat_zup]}")
        # else:
        if isinstance(rotation_quat_zup, (list, tuple)) and len(rotation_quat_zup) == 4:
            print(f"  Using original quaternion (xyzw) from JSON: {[f'{x:.4f}' for x in rotation_quat_zup]}")
        else:
            print(f"  Warning: Invalid 'rotation' format in original JSON ({rotation_quat_zup}). Using no rotation.")
            rotation_quat_zup = [0.0, 0.0, 0.0, 1.0] # xyzw

        # Check for required information (logic remains the same)
        if not scale_factor or len(scale_factor) != 3:
            print(f"  Error: Object '{instance_id}' missing valid target size information.")
            skipped_instances_log.append(f"{instance_id} (invalid size)")
            continue
        if not position_zup or len(position_zup) != 3:
             print(f"  Error: Object '{instance_id}' missing valid position information.")
             skipped_instances_log.append(f"{instance_id} (invalid position)")
             continue


        # Call processing function (logic remains the same)
        processed_obj = process_one_instance_bpy(
            i, selected_uid, position_zup, rotation_quat_zup, target_size_zup, model_base_path, instance_id
        )

        if processed_obj:
            processed_count += 1
            # link object to collection if using new_collection
            # new_collection.objects.link(processed_obj)
            # try: # Unlink might fail if it was never in the scene collection
            #     scene_collection.objects.unlink(processed_obj)
            # except: pass
        else:
            skipped_instances_log.append(f"{instance_id} ({selected_uid} processing failed)")


    # --- MODIFIED: Conditional Export and Render ---
    num_objects_in_scene = len([obj for obj in bpy.context.scene.objects if obj.type not in {'CAMERA', 'LIGHT'}]) # Only count non-auxiliary objects
    reconstruction_done = num_objects_in_scene > 0
    glb_saved = False
    rendered = False

    if reconstruction_done:
        print(f"\nReconstruction complete ({num_objects_in_scene} relevant objects).")

        # 1. Export GLB (if requested)
        if save_glb and output_glb_path:
            print(f"Preparing to export GLB to: {output_glb_path}")
            try:
                os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
                t0 = time.time()
                # Ensure Apply Modifiers is checked if needed implicitly or explicitly
                bpy.ops.export_scene.gltf(
                    filepath=output_glb_path,
                    export_format='GLB',
                    use_selection=False, # Export whole scene relevant objects
                    export_apply=True, # Apply transforms - IMPORTANT
                    export_lights=False,
                    export_cameras=False,
                    # Add compression and optimization options
                    export_draco_mesh_compression_enable=True,  # Enable Draco geometry compression
                    export_draco_mesh_compression_level=6,      # Compression level 0-10, 6 is balanced
                    export_draco_position_quantization=14,     # Position quantization bits
                    export_draco_normal_quantization=10,       # Normal quantization bits
                    export_draco_texcoord_quantization=12,     # Texture coordinate quantization bits
                    export_draco_color_quantization=10,        # Color quantization bits
                    export_draco_generic_quantization=12,      # Generic attribute quantization bits
                    export_extras=False,  
                )
                print(f"Scene successfully saved to GLB: {output_glb_path}")
                print(f"Export time: {time.time() - t0:.2f} seconds")
                glb_saved = True
            except Exception as e:
                import traceback
                print(f"\nError exporting scene to GLB: {e}")
                print(traceback.format_exc())
                glb_saved = False
        elif save_glb and not output_glb_path:
             print("Warning: GLB saving requested but no valid output_glb_path provided. Skipping GLB save.")
        else:
             print("GLB saving disabled or not requested.")

        # 2. Render views (if requested)
        if render_views and render_output_dir:
            print(f"Preparing to render views to: {render_output_dir}")
            try:
                os.makedirs(render_output_dir, exist_ok=True) # Ensure dir exists
                render_four_views(render_output_dir) # Pass the specific render directory
                rendered = True
            except Exception as e:
                 import traceback
                 print(f"\nError rendering views: {e}")
                 print(traceback.format_exc())
                 rendered = False
        elif render_views and not render_output_dir:
            print("Warning: View rendering requested but no valid render_output_dir provided. Skipping rendering.")
        else:
            print("View rendering disabled or not requested.")

    else: # reconstruction_done is False
        print("\nScene is empty or reconstruction failed, no export or rendering performed.")

    # Log skipped instances
    if skipped_instances_log:
        print("\nSkipped instances:")
        for log in skipped_instances_log:
            print(f"- {log}")

    # Return a boolean indicating whether reconstruction (placing objects) was at least partially successful
    return reconstruction_done # Or maybe return rendered status if that's the main goal?

# --- Blender script entry point ---
def main():
    """Entry point for Blender script"""
    # Parse arguments after '--' from Blender command line
    argv = sys.argv
    try:
        index = argv.index("--") + 1
        args_list = argv[index:]
    except ValueError:
        args_list = []

    parser = argparse.ArgumentParser(description="Reconstruct 3D scene from JSON using Blender")
    parser.add_argument("--scene_json", type=str, required=True, help="Path to original scene description JSON file")
    # --- REMOVED results_json ---
    # parser.add_argument("--results_json", type=str, required=True, help="Path to retrieval results JSON file")
    parser.add_argument("--model_base_path", type=str, required=True, help="Base path for 3D models (containing .glb files)")
    # --- MODIFIED output_glb ---
    parser.add_argument("--output_glb", type=str, help="Path for output GLB file (only effective when --save_glb is set)")
    # --- ADDED render_output_dir ---
    parser.add_argument("--render_output_dir", type=str, help="Output directory for rendered images (only effective when --render_views is set)")
    # --- Control flags remain ---
    parser.add_argument("--save_glb", action='store_true', help="Whether to save the output GLB file")
    parser.add_argument("--render_views", action='store_true', help="Whether to render four views")
    parser.add_argument("--table_path", type=str, help="Path to table model GLB file")
    parser.add_argument("--sink_path", type=str, help="Path to sink model GLB file")


    # Use parse_args to parse parameters from command line
    args = parser.parse_args(args_list)

    # --- Parameter validation ---
    if args.save_glb and not args.output_glb:
         parser.error("--output_glb must be provided when --save_glb is set")
    # --- MODIFIED render view check ---
    if args.render_views and not args.render_output_dir:
         parser.error("--render_output_dir must be provided when --render_views is set")

    print("="*40)
    print("Starting Blender Scene Reconstruction")
    print(f"Scene JSON: {args.scene_json}")
    # print(f"Results JSON: {args.results_json}") # Removed
    print(f"Model Path: {args.model_base_path}")
    if args.save_glb: print(f"Output GLB: {args.output_glb}")
    else: print("Output GLB: Disabled")
    if args.render_views: print(f"Render Views to: {args.render_output_dir}")
    else: print("Render Views: Disabled")
    print("="*40)


    t_start = time.time()

    # Call the main reconstruction function
    success = reconstruct_scene_from_json_bpy(
        args.scene_json, args.model_base_path,
        output_glb_path=args.output_glb,
        render_output_dir=args.render_output_dir,
        save_glb=args.save_glb,
        render_views=args.render_views,
        table_path=args.table_path,
        sink_path=args.sink_path
    )

    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.2f} seconds")
    print(f"Blender script completed. Status: {'Success' if success else 'Failed or Empty Scene'}")

if __name__ == "__main__":
    main()
