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
import argparse

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

# Add paths before importing
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Apply torchvision fix before imports
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    logger.warning("torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    logger.warning(f"Failed to apply torchvision fix: {e}")

# ——— Initialize gradio_app environment manually —————————————————————————————

logger.info("Setting up Hunyuan3D environment...")

# Create args namespace that gradio_app expects
args = argparse.Namespace(
    model_path='tencent/Hunyuan3D-2.1',
    subfolder='hunyuan3d-dit-v2-1',
    texgen_model_path='tencent/Hunyuan3D-2.1',
    port=8080,
    host='0.0.0.0',
    device='cuda',
    mc_algo='mc',
    cache_path='./temp_cache',
    enable_t23d=False,
    disable_tex=False,
    enable_flashvdm=False,
    compile=False,
    low_vram_mode=False
)

# Set required global variables BEFORE importing gradio_app
import gradio_app
gradio_app.args = args
gradio_app.SAVE_DIR = args.cache_path
os.makedirs(gradio_app.SAVE_DIR, exist_ok=True)
gradio_app.CURRENT_DIR = os.path.dirname(os.path.abspath(gradio_app.__file__))
gradio_app.MV_MODE = 'mv' in args.model_path
gradio_app.TURBO_MODE = 'turbo' in args.subfolder
gradio_app.HTML_HEIGHT = 690 if gradio_app.MV_MODE else 650
gradio_app.HTML_WIDTH = 500
gradio_app.MAX_SEED = 1e7
gradio_app.HAS_TEXTUREGEN = False

# Initialize workers and pipelines
logger.info("Initializing Hunyuan3D workers...")

# Import required modules for workers
from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

# Initialize all workers
gradio_app.rmbg_worker = BackgroundRemover()
gradio_app.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    args.model_path,
    subfolder=args.subfolder,
    use_safetensors=False,
    device=args.device,
)
gradio_app.floater_remove_worker = FloaterRemover()
gradio_app.degenerate_face_remove_worker = DegenerateFaceRemover()
gradio_app.face_reduce_worker = FaceReducer()

# Initialize texture pipeline
if not args.disable_tex:
    try:
        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        gradio_app.tex_pipeline = Hunyuan3DPaintPipeline(conf)
        gradio_app.HAS_TEXTUREGEN = True
        logger.info("Texture generation pipeline loaded")
    except Exception as e:
        logger.error(f"Failed to load texture generator: {e}")
        gradio_app.HAS_TEXTUREGEN = False

# Import the functions we need
from gradio_app import generation_all, gen_save_folder, export_mesh, randomize_seed_fn, export_to_trimesh, quick_convert_with_obj2gltf

logger.info("Successfully initialized Hunyuan3D environment")

# ——— Utility functions —————————————————————————————————————————————————————

def clear_memory():
    """Clear GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def cleanup_temp_folders(base_save_dir):
    """Clean up temporary folders created by gen_save_folder."""
    if os.path.exists(base_save_dir):
        folders = sorted(
            [f for f in Path(base_save_dir).iterdir() if f.is_dir()],
            key=lambda x: x.stat().st_mtime
        )
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
        
        # Call generation_all - this handles everything
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
            seed=1234,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True
        )
        
        # Unpack the results
        file_out = result[0]
        file_out2 = result[1]
        html_output = result[2]
        stats = result[3]
        seed = result[4]
        
        logger.info(f"Generation complete. Seed used: {seed}")
        logger.info(f"Stats: {stats.get('number_of_faces', 'N/A')} faces, {stats.get('number_of_vertices', 'N/A')} vertices")
        
        # Extract the GLB path
        if hasattr(file_out2, 'value'):
            glb_path = file_out2.value
        else:
            glb_path = file_out2
            
        logger.info(f"Textured GLB at: {glb_path}")
        
        # Move to output directory
        final_path = os.path.join(output_dir, f"{base_name}_textured.glb")
        shutil.copy2(glb_path, final_path)
        logger.info(f"Saved final model: {final_path}")
        
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
    input_dir = "/data/in"
    output_dir = "/data/out"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./temp_cache', exist_ok=True)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    
    if not images:
        logger.warning("No PNG images found in input directory")
        return
    
    logger.info(f"Found {len(images)} images to process")
    
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
        
        cleanup_temp_folders('./temp_cache')
        cleanup_output_dir(base_name, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Successful: {success_count}/{len(images)}")
    if failed_images:
        logger.info(f"Failed: {failed_images}")
    logger.info(f"{'='*60}")
    
    try:
        shutil.rmtree('./temp_cache')
        logger.info("Cleaned up temporary cache directory")
    except:
        pass

if __name__ == "__main__":
    main()
