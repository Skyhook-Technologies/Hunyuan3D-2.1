import os
import sys
import gc
import glob
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from PIL import Image
from tqdm import tqdm

# Add Hunyuan 3D directories to Python path
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Hunyuan3D-2.1 pipeline imports
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# Apply torchvision compatibility fix if available
try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("Info: torchvision compatibility fix applied successfully.")
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix.")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# Memory management function
def clear_memory():
    """Clear GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Enhanced cleanup function
def cleanup_intermediate_files(base_name, output_dir):
    """Remove all intermediate files except the final textured GLB"""
    final_output = os.path.join(output_dir, base_name + "_textured.glb")
    
    # Patterns of intermediate files to remove
    patterns_to_remove = [
        f"{base_name}_mesh.glb",           # Untextured mesh
        f"{base_name}_textured.jpg",       # Texture image
        f"{base_name}_textured.mtl",       # Material file
        f"{base_name}_textured_metallic.jpg",  # Metallic map
        f"{base_name}_textured_roughness.jpg", # Roughness map
        "white_mesh_remesh.obj",           # Remeshed geometry
        f"{base_name}_*.obj",              # Any obj files
        f"{base_name}_*.jpg",              # Any remaining jpg files (except final if needed)
        f"{base_name}_*.mtl",              # Any remaining mtl files
        f"{base_name}_*.png",              # Any intermediate png files
    ]
    
    files_removed = []
    for pattern in patterns_to_remove:
        pattern_path = os.path.join(output_dir, pattern)
        matching_files = glob.glob(pattern_path)
        
        for file_path in matching_files:
            # Don't remove the final textured GLB file
            if file_path != final_output:
                try:
                    os.remove(file_path)
                    files_removed.append(os.path.basename(file_path))
                except Exception as e:
                    print(f"Warning: Failed to remove {file_path}: {e}")
    
    # Also check for any files that might be created in current directory
    current_dir_patterns = [
        "white_mesh_remesh.obj",
        "*.tmp",
        "temp_*"
    ]
    
    for pattern in current_dir_patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                os.remove(file_path)
                files_removed.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Warning: Failed to remove {file_path}: {e}")
    
    if files_removed:
        print(f"Info: Cleaned up intermediate files: {', '.join(files_removed)}")
    else:
        print("Info: No intermediate files found to clean up")

# --- Initialize Pipelines ---
print("Info: Initializing Hunyuan 3D pipelines...")

# Shape generation pipeline
model_path = 'tencent/Hunyuan3D-2.1'
try:
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    print("Info: Shape generation pipeline loaded.")
except Exception as e:
    print(f"Error: Failed to load shape generation pipeline. Check model path and dependencies. {e}")
    sys.exit(1)

# Background remover (instantiated once for efficiency)
rembg_processor = BackgroundRemover()
print("Info: Background remover initialized.")

# Texture generation pipeline config
max_num_view = 6
resolution = 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
# Note: These paths are hardcoded. Ensure the repo is cloned with this structure.
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

# Texture generation pipeline
try:
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    print("Info: Texture generation pipeline loaded.")
except Exception as e:
    print(f"Error: Failed to load texture generation pipeline. Check config and file paths. {e}")
    sys.exit(1)

# Clear memory after initialization
clear_memory()

# --- Define I/O Directories ---
input_dir = "/data/in"
output_dir = "/data/out"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Info: Input directory is '{input_dir}'. Output directory is '{output_dir}'.")

# List all PNG files in the input directory
if not os.path.exists(input_dir):
    print(f"Error: Input directory '{input_dir}' not found. Please ensure images are downloaded.")
    sys.exit(1)

images = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
images.sort()  # Sort for consistent processing order

if not images:
    print(f"Warning: No PNG images found in '{input_dir}'. Script will terminate.")
    sys.exit(0)

# --- Process Images in Batch ---
print(f"Info: Found {len(images)} images to process.")

for img_name in tqdm(images, desc="Processing images"):
    img_path = os.path.join(input_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    output_path = os.path.join(output_dir, base_name + "_textured.glb")
    temp_mesh_path = os.path.join(output_dir, base_name + "_mesh.glb")

    print(f"\n--- Starting processing for {img_name} ---")
    
    try:
        # Load image and convert to RGBA
        image = Image.open(img_path).convert("RGBA")
        
        # FIXED: Remove background if image is RGB (has no transparency)
        # The original logic was backwards - we need to check the original mode before conversion
        original_image = Image.open(img_path)
        if original_image.mode == 'RGB':
            print(f"Info: Removing background from {img_name}...")
            image = rembg_processor(image)
        
        # Clear memory after background removal
        clear_memory()

        # Stage 1: Generate 3D mesh (untextured)
        print(f"Info: Generating 3D mesh from {img_name}...")
        mesh = shape_pipeline(image=image)[0]
        mesh.export(temp_mesh_path)
        print(f"Info: Untextured mesh saved to {temp_mesh_path}")
        
        # Clear memory after mesh generation
        del mesh
        clear_memory()

        # Stage 2: Apply texture to the mesh
        print(f"Info: Applying texture to the mesh...")
        paint_pipeline(
            mesh_path=temp_mesh_path,
            image_path=img_path,
            output_mesh_path=output_path
        )
        print(f"Info: Textured mesh saved to {output_path}")

        # Clear memory after texture generation
        clear_memory()

        print(f"Info: Processing completed successfully for {img_name}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed processing {img_name}: {e}")
        # Clear memory even on error
        clear_memory()
        # Log the error and move to the next image
    
    finally:
        # ENHANCED: Clean up ALL intermediate files, not just the temp mesh
        cleanup_intermediate_files(base_name, output_dir)
        
        # Clean up image variables if they exist
        try:
            if 'image' in locals():
                del image
            if 'original_image' in locals():
                del original_image
        except:
            pass
        
        # Final memory cleanup
        clear_memory()

print("\nAll images processed. Script finished.")
