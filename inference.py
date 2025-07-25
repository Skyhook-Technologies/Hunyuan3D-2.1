import os
from PIL import Image
from tqdm import tqdm

# Hunyuan3D-2.1 pipeline imports
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # shape generator
from hy3dshape.rembg import BackgroundRemover                    # background removal
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig  # texture generator

# Initialize pipelines once (reuse for all images to avoid re-loading models repeatedly)
model_path = 'tencent/Hunyuan3D-2.1'  # Hugging Face model repo for shape&texture
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)  # load shape model:contentReference[oaicite:4]{index=4}

# Configure texture generation pipeline
max_num_view = 6   # number of views to sample (6-9 recommended)
resolution = 512   # texture resolution (512 or 768 supported)
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
# Specify required paths for texture pipeline (weights, config)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"               # super-resolution weights:contentReference[oaicite:5]{index=5}
conf.multiview_cfg_path  = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"               # model config for texture
conf.custom_pipeline    = "hy3dpaint/hunyuanpaintpbr"                           # custom pipeline code path
paint_pipeline = Hunyuan3DPaintPipeline(conf)  # initialize texture model pipeline

# Create output directory if not exists
os.makedirs("/data/out", exist_ok=True)

# List all PNG files in input directory
input_dir = "/data/in"
images = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
images.sort()  # sort for consistent order (optional)

for img_name in tqdm(images, desc="Processing images"):
    img_path = os.path.join(input_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    output_path = os.path.join("/data/out", base_name + ".glb")
    try:
        # Load image and add alpha channel if needed
        image = Image.open(img_path).convert("RGBA")
        if image.mode == "RGB":
            # Remove background for better mesh generation (if no alpha present)
            image = BackgroundRemover()(image)  # apply background removal:contentReference[oaicite:6]{index=6}

        # Stage 1: Generate 3D mesh (untextured) from the image
        mesh = shape_pipeline(image=image)[0]           # get predicted mesh (trimesh object):contentReference[oaicite:7]{index=7}
        temp_mesh_path = os.path.join("/data/out", base_name + "_mesh.glb")
        mesh.export(temp_mesh_path)                     # save intermediate mesh to GLB file

        # Stage 2: Apply texture to the mesh using the same image
        paint_pipeline(
            mesh_path=temp_mesh_path, 
            image_path=img_path, 
            output_mesh_path=output_path
        )
        # If successful, remove the intermediate mesh file to save space (optional):
        os.remove(temp_mesh_path)
    except Exception as e:
        print(f"[Error] Failed processing {img_name}: {e}")
