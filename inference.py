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

# Ensure expandable segments for CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add Hunyuan3D code paths
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Core imports
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

# Apply torchvision compatibility fix
try:
    from torchvision_fix import apply_fix
    apply_fix()
    print("Info: torchvision compatibility fix applied successfully.")
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix.")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

# ——— Utility functions —————————————————————————————————————————————————————

def clear_memory():
    """Clear GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def cleanup_intermediate_files(base_name, output_dir):
    """Remove all intermediate files except the final textured GLB."""
    final_output = os.path.join(output_dir, base_name + "_textured.glb")
    patterns = [
        f"{base_name}_mesh.glb",
        f"{base_name}_mesh.obj",
        f"{base_name}_textured.obj",
        f"{base_name}_textured.jpg",
        f"{base_name}_textured.mtl",
        f"{base_name}_textured_metallic.jpg",
        f"{base_name}_textured_roughness.jpg",
        f"{base_name}_*.obj",
        f"{base_name}_*.jpg",
        f"{base_name}_*.mtl",
        f"{base_name}_*.png",
        "white_mesh_remesh.obj",
        "*.tmp",
        "temp_*"
    ]
    removed = []
    for pat in patterns:
        for fpath in glob.glob(os.path.join(output_dir, pat)):
            if os.path.abspath(fpath) != os.path.abspath(final_output):
                try:
                    os.remove(fpath)
                    removed.append(os.path.basename(fpath))
                except:
                    pass
    if removed:
        print(f"Info: Cleaned up intermediates: {', '.join(removed)}")

# ——— Initialize pipelines —————————————————————————————————————————————————————

print("Info: Initializing Hunyuan 3D pipelines...")

# Shape generation pipeline
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    subfolder='hunyuan3d-dit-v2-1',  # CRITICAL
    use_safetensors=False,
    device='cuda'
)
print("Info: Shape generation pipeline loaded.")

# Background remover
rembg_processor = BackgroundRemover()
print("Info: Background remover initialized.")

# Post-processing workers
floater_remove_worker       = FloaterRemover()
degenerate_face_remove_worker = DegenerateFaceRemover()
face_reduce_worker          = FaceReducer()
print("Info: Post-processing workers initialized.")

# Texture generation pipeline (6 views, 512 res)
conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path   = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline      = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)
print("Info: Texture generation pipeline loaded.")

clear_memory()

# ——— I/O setup ——————————————————————————————————————————————————————————————

input_dir  = "/data/in"
output_dir = "/data/out"
os.makedirs(output_dir, exist_ok=True)
print(f"Info: Input directory: {input_dir}")
print(f"Info: Output directory: {output_dir}")

# Collect PNGs
images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
if not images:
    print("Warning: No PNGs found in input directory; exiting.")
    sys.exit(0)

# ——— Batch processing loop —————————————————————————————————————————————————————

MAX_SEED = int(1e7)
for img_name in tqdm(images, desc="Processing images"):
    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(input_dir, img_name)
    temp_obj     = os.path.join(output_dir, f"{base}_mesh.obj")
    textured_obj = os.path.join(output_dir, f"{base}_textured.obj")
    final_glb    = os.path.join(output_dir, f"{base}_textured.glb")

    print(f"\n--- Processing {img_name} ---")
    try:
        # 1) Load & remove background if needed
        orig = Image.open(img_path)
        if orig.mode == "RGB":
            print("Info: Removing background")
            img = rembg_processor(orig.convert("RGB"))
        else:
            img = orig.convert("RGBA")
        clear_memory()

        # 2) Shape generation
        seed = random.randint(0, MAX_SEED)
        gen = torch.Generator().manual_seed(seed)
        print(f"Info: Generating mesh (seed={seed})")
        outputs = shape_pipeline(
            image=img,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=gen,
            octree_resolution=256,
            num_chunks=8000,
            output_type='mesh'
        )
        mesh = export_to_trimesh(outputs)[0]
        print(f"Info: Mesh faces={mesh.faces.shape[0]}, verts={mesh.vertices.shape[0]}")
        clear_memory()

        # FIXED: Apply floater and degenerate face removal BEFORE face reduction and texturing.
        # This ensures the texture is applied to a cleaner mesh.
        print("Info: Initial mesh cleanup (floater and degenerate faces)")
        mesh = floater_remove_worker(mesh)
        mesh = degenerate_face_remove_worker(mesh)
        clear_memory()

        # 3) Face reduction & OBJ export
        print("Info: Reducing faces")
        mesh = face_reduce_worker(mesh)  # default ~10000 faces
        mesh.export(temp_obj, include_normals=True)
        clear_memory()

        # 4) Texture painting → OBJ
        print("Info: Painting texture")
        paint_pipeline(
            mesh_path=temp_obj,
            image_path=img_path,
            output_mesh_path=textured_obj,
            save_glb=False
        )
        clear_memory()

        # 5) OBJ → GLB with PBR
        print("Info: Converting OBJ to GLB")
        textures = {
            'albedo':    textured_obj.replace('.obj', '.jpg'),
            'metallic':  textured_obj.replace('.obj', '_metallic.jpg'),
            'roughness': textured_obj.replace('.obj', '_roughness.jpg'),
        }
        create_glb_with_pbr_materials(textured_obj, textures, final_glb)
        clear_memory()

        # Removed the final floater/degenerate cleanup as it's now done earlier.
        # The GLB is now directly saved.

        print(f"Saved: {final_glb}")

    except Exception as e:
        print(f"[ERROR] {base}: {e}")

    finally:
        cleanup_intermediate_files(base, output_dir)
        clear_memory()

print("\nAll done.")
