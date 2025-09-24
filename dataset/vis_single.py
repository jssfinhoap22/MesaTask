#!/usr/bin/env python3
"""
Script for visualizing a single layout
"""

import os
import sys
import json
import argparse
import subprocess
import yaml
from pathlib import Path

def load_config(config_path=None):
    """Load configuration file"""
    if not config_path:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def visualize_layout(layout_json_path, output_dir=None, config_path=None, save_glb=True, render_views=True):
    """Visualize layout"""
    layout_path = Path(layout_json_path)
    if not layout_path.exists():
        raise FileNotFoundError(f"File not found: {layout_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set output directory - use parent directory name as output folder name
    scene_name = layout_path.parent.name
    output_dir = Path(output_dir) / f'{scene_name}_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find reconstruct_blender.py script
    script_dir = Path(__file__).parent
    reconstruct_script = script_dir / "reconstruct_blender.py"
    if not reconstruct_script.exists():
        raise FileNotFoundError(f"Script not found: {reconstruct_script}")
    
    # Build command
    cmd = [
        config['blender_executable'],
        "--background",
        "--python", str(reconstruct_script),
        "--",
        "--scene_json", str(layout_path),
        "--model_base_path", config['model_base_path']
    ]
    
    # Add table and sink model paths
    if 'table_model_path' in config:
        cmd.extend(["--table_path", config['table_model_path']])
    
    if 'sink_model_path' in config:
        cmd.extend(["--sink_path", config['sink_model_path']])
    
    if save_glb:
        cmd.extend(["--save_glb", "--output_glb", str(output_dir / f"{layout_path.stem}.glb")])
    
    if render_views:
        cmd.extend(["--render_views", "--render_output_dir", str(output_dir / "rendered_views")])
    
    # Execute
    print(f"Visualizing: {layout_path.name}")
    print(f"Output to: {output_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    
    if result.returncode == 0:
        print("Completed!")
        return True
    else:
        print(f"Failed: {result.stderr}")
        return False

def show_info(layout_json_path):
    """Display layout information"""
    try:
        with open(layout_json_path, 'r') as f:
            data = json.load(f)
        
        objects = data.get('objects', [])
        print(f"\n {Path(layout_json_path).name}")
        print(f"Number of objects: {len(objects)}")
        
        for i, obj in enumerate(objects[:5]):  # Only show first 5 objects
            instance = obj.get('instance', 'unknown')
            uid = obj.get('retrieved_uid', 'unknown')
            print(f"  {i+1}. {instance} -> {uid}")
        
        if len(objects) > 5:
            print(f"  ... and {len(objects)-5} more objects")
        print()
    except Exception as e:
        print(f"Failed to read file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Single layout visualization")
    parser.add_argument("layout_json", help="Path to layout JSON file")
    parser.add_argument("--output_dir", 
                       default='./vis_dataset',
                       help="Output directory")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--no_glb", action='store_true', help="Don't save GLB")
    parser.add_argument("--no_render", action='store_true', help="Don't render views")
    parser.add_argument("--info_only", action='store_true', help="Show info only")
    
    args = parser.parse_args()
    
    # Show information
    show_info(args.layout_json)
    
    if args.info_only:
        return
    
    # Visualize
    try:
        success = visualize_layout(
            args.layout_json,
            args.output_dir,
            args.config,
            not args.no_glb,
            not args.no_render
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
