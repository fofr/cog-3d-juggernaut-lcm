# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import time
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Below is an example showing how to get the node you need and update the inputs

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = kwargs["cfg"]
        sampler["denoise"] = kwargs["denoise"]

        checkpoint_loader = workflow["4"]["inputs"]
        checkpoint_loader["ckpt_name"] = kwargs["checkpoint"]

        lora_loader = workflow["19"]["inputs"]
        lora_loader["strength_model"] = kwargs["lora_strength"]

        lcm_lora_loader = workflow["18"]["inputs"]
        lcm_lora_loader["strength_model"] = kwargs["lcm_lora_strength"]

        control_net = workflow["15"]["inputs"]
        control_net["strength"] = kwargs["control_strength"]

        image_resizer = workflow["13"]["inputs"]
        image_resizer["width"] = kwargs["max_width"]
        image_resizer["height"] = kwargs["max_height"]

    def predict(
        self,
        prompt: str = Input(
            default="3D Render Style, 3DRenderAF, unreal engine, video game, pixelated, low poly",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="photo, photography, realistic",
        ),
        image: Path = Input(
            description="An input image",
        ),
        max_width: int = Input(
            description="The maximum width of the image",
            default=512,
        ),
        max_height: int = Input(
            description="The maximum height of the image",
            default=512,
        ),
        checkpoint: str = Input(
            description="The checkpoint to use",
            default="juggernaut_reborn.safetensors",
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
        steps: int = Input(
            description="The number of steps to take",
            default=4,
        ),
        cfg: float = Input(
            description="The CFG",
            default=1,
        ),
        denoise: float = Input(
            description="The denoise",
            default=0.65,
        ),
        lora_strength: float = Input(
            description="The strength of the lora",
            default=2,
        ),
        lcm_lora_strength: float = Input(
            description="The strength of the LCM lora",
            default=1.5,
        ),
        control_strength: float = Input(
            description="The strength of the control net",
            default=0.5,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        seed = seed_helper.generate(seed)

        start_time = time.time()
        self.handle_input_file(image)
        print(f"Time taken to handle input file: {(time.time() - start_time):.2f} seconds")

        start_time = time.time()
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        print(f"Time taken to load workflow: {(time.time() - start_time):.2f} seconds")

        start_time = time.time()
        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            checkpoint=checkpoint,
            lora_strength=lora_strength,
            lcm_lora_strength=lcm_lora_strength,
            control_strength=control_strength,
            max_width=max_width,
            max_height=max_height,
        )
        print(f"Time taken to update workflow: {(time.time() - start_time):.2f} seconds")

        start_time = time.time()
        wf = self.comfyUI.load_workflow(workflow)
        print(f"Time taken to load workflow: {(time.time() - start_time):.2f} seconds")

        start_time = time.time()
        self.comfyUI.connect()
        print(f"Time taken to connect: {(time.time() - start_time):.2f} seconds")

        start_time = time.time()
        self.comfyUI.run_workflow(wf)
        print(f"Time taken to run workflow: {(time.time() - start_time):.2f} seconds")

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
