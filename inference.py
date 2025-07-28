import os
import sys
import gc
import glob
import torch
import random
import trimesh
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add Hunyuan 3D directories to Python path
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Hunyuan3D-2.1 pipeline imports
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

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
        f"{base_name}_mesh.obj",           # Untextured obj
        f"{base_name}_textured.obj",       # Textured obj
        f"{base_name}_textured.jpg",       # Texture image
        f"{base_name}_textured.mtl",       # Material file
        f"{base_name}_textured_metallic.jpg",  # Metallic map
        f"{base_name}_textured_roughness.jpg", # Roughness map
        "white_mesh_remesh.obj",           # Remeshed geometry
        f"{base_name}_*.obj",              # Any obj files
        f"{base_name}_*.jpg",              # Any remaining jpg files
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

# Initialize post-processing workers (matching HuggingFace)
floater_remove_worker = FloaterRemover()
degenerate_face_remove_worker = DegenerateFaceRemover()
face_reduce_worker = FaceReducer()
print("Info: Post-processing workers initialized.")

# Texture generation pipeline config (matching HuggingFace settings)
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

# Maximum seed value (matching HuggingFace)
MAX_SEED = int(1e7)

for img_name in tqdm(images, desc="Processing images"):
    img_path = os.path.join(input_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    
    # File paths
    temp_mesh_obj = os.path.join(output_dir, base_name + "_mesh.obj")
    temp_textured_obj = os.path.join(output_dir, base_name + "_textured.obj")
    final_glb_path = os.path.join(output_dir, base_name + "_textured.glb")

    print(f"\n--- Starting processing for {img_name} ---")
    
    try:
        # Load image and convert to RGBA
        image = Image.open(img_path).convert("RGBA")
        
        # Remove background if image is RGB (matching HuggingFace logic)
        original_image = Image.open(img_path)
        if original_image.mode == 'RGB':
            print(f"Info: Removing background from {img_name}...")
            image = rembg_processor(image)
        
        # Clear memory after background removal
        clear_memory()

        # Stage 1: Generate 3D mesh (matching HuggingFace generation_all function)
        print(f"Info: Generating 3D mesh from {img_name}...")
        
        # Randomize seed for each generation
        seed = random.randint(0, MAX_SEED)
        generator = torch.Generator()
        generator = generator.manual_seed(seed)
        
        # Generate mesh with exact HuggingFace parameters
        outputs = shape_pipeline(
            image=image,
            num_inference_steps=30,      # HuggingFace: steps=30
            guidance_scale=5.0,          # HuggingFace: guidance_scale=5
            generator=generator,
            octree_resolution=256,       # HuggingFace: octree_resolution=256
            num_chunks=8000,            # HuggingFace: num_chunks=8000
            output_type='mesh'
        )
        
        # Export to trimesh (following HuggingFace workflow)
        mesh = export_to_trimesh(outputs)[0]
        
        print(f"Info: Generated mesh with seed {seed}")
        print(f"Info: Mesh stats - Faces: {mesh.faces.shape[0]}, Vertices: {mesh.vertices.shape[0]}")
        
        # Clear memory after mesh generation
        del outputs
        clear_memory()
        
        # Stage 2: Post-processing (matching HuggingFace workflow)
        print("Info: Post-processing mesh...")
        
        # Note: HuggingFace comments out floater and degenerate face removal
        # mesh = floater_remove_worker(mesh)
        # mesh = degenerate_face_remove_worker(mesh)
        
        # Face reduction (HuggingFace uses default 10000 faces)
        print("Info: Reducing mesh faces...")
        mesh = face_reduce_worker(mesh, 10000)
        print(f"Info: Reduced mesh - Faces: {mesh.faces.shape[0]}, Vertices: {mesh.vertices.shape[0]}")
        
        # Export as OBJ for texture painting (HuggingFace exports as 'obj' before texturing)
        mesh.export(temp_mesh_obj, include_normals=False)
        
        # Clear memory
        del mesh
        clear_memory()

        # Stage 3: Apply texture (matching HuggingFace tex_pipeline call)
        print(f"Info: Applying texture to the mesh...")
        paint_pipeline(
            mesh_path=temp_mesh_obj,
            image_path=img_path,
            output_mesh_path=temp_textured_obj,
            save_glb=False  # Important: save as OBJ first like HuggingFace
        )
        
        print("Info: Texture generation completed")
        
        # Clear memory after texture generation
        clear_memory()

        # Stage 4: Convert textured OBJ to GLB with PBR materials (matching HuggingFace)
        print("Info: Converting to GLB with PBR materials...")
        
        # This matches the quick_convert_with_obj2gltf function in HuggingFace
        textures = {
            'albedo': temp_textured_obj.replace('.obj', '.jpg'),
            'metallic': temp_textured_obj.replace('.obj', '_metallic.jpg'),
            'roughness': temp_textured_obj.replace('.obj', '_roughness.jpg')
        }
        
        # This function handles the coordinate system conversion properly
        create_glb_with_pbr_materials(temp_textured_obj, textures, final_glb_path)
        
        print(f"Info: Final GLB saved to {final_glb_path}")
        print(f"Info: Processing completed successfully for {img_name}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed processing {img_name}: {e}")
        import traceback
        traceback.print_exc()
        # Clear memory even on error
        clear_memory()
        # Log the error and move to the next image
    
    finally:
        # Clean up ALL intermediate files
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
