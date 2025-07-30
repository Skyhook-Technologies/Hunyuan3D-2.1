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
import gradio as gr

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

# Make export_to_trimesh available in gradio_app module
gradio_app.export_to_trimesh = export_to_trimesh

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

# Import the functions we need from gradio_app
from gradio_app import generation_all, gen_save_folder, export_mesh, randomize_seed_fn, quick_convert_with_obj2gltf

# Import and set up the on_export_click function from build_app
gradio_app_module = sys.modules['gradio_app']
exec(open('gradio_app.py').read(), gradio_app_module.__dict__)

# Extract on_export_click from the module after executing
def get_on_export_click():
    """Extract the on_export_click function from gradio_app's build_app"""
    # We need to access the function defined inside build_app
    # Since it's a nested function, we'll need to replicate its logic
    def on_export_click(file_out, file_out2, file_type, reduce_face, export_texture, target_face_num):
        if file_out is None:
            raise gr.Error('Please generate a mesh first.')

        print(f'exporting {file_out}')
        print(f'reduce face to {target_face_num}')
        if export_texture:
            mesh = trimesh.load(file_out2)
            save_folder = gen_save_folder()
            path = export_mesh(mesh, save_folder, textured=True, type=file_type)

            # for preview
            save_folder = gen_save_folder()
            _ = export_mesh(mesh, save_folder, textured=True)
            model_viewer_html = gradio_app.build_model_viewer_html(save_folder, 
                                                        height=gradio_app.HTML_HEIGHT, 
                                                        width=gradio_app.HTML_WIDTH,
                                                        textured=True)
        else:
            mesh = trimesh.load(file_out)
            mesh = gradio_app.floater_remove_worker(mesh)
            mesh = gradio_app.degenerate_face_remove_worker(mesh)
            if reduce_face:
                mesh = gradio_app.face_reduce_worker(mesh, target_face_num)
            save_folder = gen_save_folder()
            path = export_mesh(mesh, save_folder, textured=False, type=file_type)

            # for preview
            save_folder = gen_save_folder()
            _ = export_mesh(mesh, save_folder, textured=False)
            model_viewer_html = gradio_app.build_model_viewer_html(save_folder, 
                                                        height=gradio_app.HTML_HEIGHT, 
                                                        width=gradio_app.HTML_WIDTH,
                                                        textured=False)
        print(f'export to {path}')
        return model_viewer_html, gr.update(value=path, interactive=True)
    
    return on_export_click

on_export_click = get_on_export_click()

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
    """Process a single image through the Hunyuan3D pipeline - matching HF API exactly."""
    try:
        # Load image
        logger.info(f"Loading image: {img_path}")
        image = Image.open(img_path).convert('RGBA')
        
        # Step 1: Call generation_all (matching HF API first call)
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
        
        # Extract actual file paths from gr.update objects
        if isinstance(file_out, dict) and 'value' in file_out:
            file_out_path = file_out['value']
        else:
            file_out_path = file_out
            
        if isinstance(file_out2, dict) and 'value' in file_out2:
            file_out2_path = file_out2['value']
        else:
            file_out2_path = file_out2
        
        # Step 2: Call on_export_click (matching HF API second call)
        logger.info("Exporting with texture...")
        export_html, export_result = on_export_click(
            file_out=file_out_path,
            file_out2=file_out2_path,
            file_type="glb",
            reduce_face=True,
            export_texture=True,
            target_face_num=10000
        )
        
        # Extract the exported file path
        if isinstance(export_result, dict) and 'value' in export_result:
            exported_file_path = export_result['value']
        else:
            exported_file_path = export_result
            
        logger.info(f"Export complete: {exported_file_path}")
        
        # Move to output directory
        final_path = os.path.join(output_dir, f"{base_name}_textured.glb")
        shutil.copy2(exported_file_path, final_path)
        logger.info(f"Saved final model: {final_path}")
        
        # Clean up intermediate files (matching HF cleanup)
        for cleanup_path in [file_out_path, file_out2_path]:
            if os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                except:
                    pass
        
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
