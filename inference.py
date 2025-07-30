import os
import sys
import gc
import shutil
import torch
import random
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import glob
import trimesh

# Ensure expandable segments for CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ——— Setup gradio_app environment —————————————————————————————————————————————

# CRITICAL: Set up args BEFORE importing gradio_app
sys.argv = [
    'gradio_app.py',
    '--model_path', 'tencent/Hunyuan3D-2.1',
    '--subfolder', 'hunyuan3d-dit-v2-1',
    '--texgen_model_path', 'tencent/Hunyuan3D-2.1',
    '--cache-path', './temp_cache',  # Temporary cache directory
    '--device', 'cuda',
    '--mc_algo', 'mc',
]

logger.info("Initializing Hunyuan3D through gradio_app...")

# Add paths before importing
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Import and run gradio_app initialization
import gradio_app

# Now we need to trigger the initialization that happens in if __name__ == '__main__':
# We'll access the initialized variables through the gradio_app module
logger.info("Waiting for gradio_app initialization...")

# The workers are initialized in the main block, so we need to access them after import
# Import the functions we can access
from gradio_app import generation_all, shape_generation, gen_save_folder, export_mesh

# Also import the conversion function
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

logger.info("Successfully imported core functions")

# ——— Utility functions —————————————————————————————————————————————————————

def clear_memory():
    """Clear GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def cleanup_temp_folders(base_save_dir):
    """Clean up temporary folders created by gen_save_folder."""
    if os.path.exists(base_save_dir):
        # Keep only the most recent folders to prevent disk fill
        folders = sorted(
            [f for f in Path(base_save_dir).iterdir() if f.is_dir()],
            key=lambda x: x.stat().st_mtime
        )
        # Keep last 5 folders, remove older ones
        if len(folders) > 5:
            for folder in folders[:-5]:
                try:
                    shutil.rmtree(folder)
                    logger.debug(f"Removed old temp folder: {folder}")
                except Exception as e:
                    logger.warning(f"Failed to remove {folder}: {e}")

def cleanup_output_dir(base_name, output_dir):
    """Remove any intermediate files, keeping only the final GLB."""
    final_output = os.path.join(output_dir, f"{base_name}_textured.glb")
    patterns = [
        f"{base_name}_*.obj",
        f"{base_name}_*.jpg",
        f"{base_name}_*.png",
        f"{base_name}_*.mtl",
        "temp_*",
        "*.tmp"
    ]
    
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.abspath(filepath) != os.path.abspath(final_output):
                try:
                    os.remove(filepath)
                except:
                    pass

# ——— Main processing —————————————————————————————————————————————————————

def process_image(img_path, base_name, output_dir):
    """Process a single image through the Hunyuan3D pipeline."""
    try:
        # Load image
        logger.info(f"Loading image: {img_path}")
        image = Image.open(img_path).convert('RGBA')
        
        # Call generation_all - this handles everything (shape + texture)
        logger.info(f"Generating 3D model for {base_name}...")
        result = generation_all(
            caption=None,
            image=image,
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=30,
            guidance_scale=5.0,
            seed=1234,  # Will be randomized due to randomize_seed=True
            octree_resolution=256,
            check_box_rembg=True,  # Remove background if needed
            num_chunks=8000,
            randomize_seed=True  # Ensures different seed each generation
        )
        
        # Unpack the results - generation_all returns a tuple
        file_out = result[0]
        file_out2 = result[1]
        html_output = result[2]
        stats = result[3]
        seed = result[4]
        
        logger.info(f"Shape and texture generation complete. Seed used: {seed}")
        logger.info(f"Stats: {stats.get('number_of_faces', 'N/A')} faces, {stats.get('number_of_vertices', 'N/A')} vertices")
        
        # Extract the actual file paths
        if hasattr(file_out2, 'value'):
            glb_path = file_out2.value
        else:
            glb_path = file_out2
            
        logger.info(f"Textured GLB generated at: {glb_path}")
        
        # Move the final file to output directory with our naming convention
        final_path = os.path.join(output_dir, f"{base_name}_textured.glb")
        shutil.copy2(glb_path, final_path)  # Use copy2 to preserve metadata
        logger.info(f"Saved final model: {final_path}")
        
        # Clear memory after each generation
        clear_memory()
        
        return True, final_path
        
    except Exception as e:
        logger.error(f"Error processing {base_name}: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()
        return False, None

# ——— Main execution —————————————————————————————————————————————————————

def main():
    # I/O directories
    input_dir = "/data/in"
    output_dir = "/data/out"
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./temp_cache', exist_ok=True)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Collect PNG images
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    
    if not images:
        logger.warning("No PNG images found in input directory")
        return
    
    logger.info(f"Found {len(images)} images to process")
    
    # Process each image
    success_count = 0
    failed_images = []
    
    for img_name in tqdm(images, desc="Processing images"):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(input_dir, img_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {img_name} ({images.index(img_name)+1}/{len(images)})")
        logger.info(f"{'='*60}")
        
        success, final_path = process_image(img_path, base_name, output_dir)
        
        if success:
            success_count += 1
        else:
            failed_images.append(img_name)
        
        # Clean up temporary folders periodically
        cleanup_temp_folders('./temp_cache')
        
        # Clean up any intermediate files in output dir
        cleanup_output_dir(base_name, output_dir)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Successful: {success_count}/{len(images)}")
    if failed_images:
        logger.info(f"Failed: {failed_images}")
    logger.info(f"{'='*60}")
    
    # Final cleanup of temp cache
    try:
        shutil.rmtree('./temp_cache')
        logger.info("Cleaned up temporary cache directory")
    except:
        pass

if __name__ == "__main__":
    main()
