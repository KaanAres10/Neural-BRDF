# SPDX-License-Identifier: Apache-2.0
from slangpy import Device, DeviceType, TextureLoader, Module, Timer

from app import App
import slangpy as spy
from slangpy.types import NDBuffer, call_id
import numpy as np
import math
import time
import os

import neuralnetworks as nn

from slangpy.reflection import SlangType


class Timer:
    def __init__(self, history: int = 16):
        super().__init__()
        self.index = 0
        self.begin = None
        self.times = [0.0] * history
        self.history = history

    def start(self):
        self.begin = time.time()

    def stop(self):
        if self.begin is None:
            return

        t = time.time()
        elapsed = t - self.begin
        self.begin = t

        self.times[self.index % self.history] = elapsed
        self.index += 1

        return self.elapsed()

    def elapsed(self):
        l = min(self.index, self.history)
        return 0 if l == 0 else sum(self.times[:l]) / l

    def frequency(self):
        e = self.elapsed()
        return 0 if e == 0 else 1.0 / e


resolution = 720


# This Camera class mimics the Camera struct from brdf.slang, setting
# the origin, scale, and window size that the Slang camera struct expects. The
# window size is retrieved from the app window to avoid redundant state.
class Camera:
    # Origin is at 0,0 with 1,1 scale.
    def __init__(self, app: App):
        super().__init__()
        self.o = spy.float2(0.0, 0.0)
        self.scale = spy.float2(1.0, 1.0)
        self.app = app

    # Return a dict with the class variables mapped to the names that the
    # Slang struct expects.
    def get_this(self):
        return {
            "o": self.o,
            "scale": self.scale,
            "frameDim": spy.float2(resolution, resolution),
            "_type": "Camera",
        }


# This Properties class mimics the Properties struct from brdf.slang,
# setting the baseColor, roughness, metallic, and specular values that the
# Properties struct expects.


class Properties:
    def __init__(self, b: spy.float3, r: float, m: float, s: float):
        super().__init__()
        self.b = b
        self.r = r
        self.m = m
        self.s = s

    # Return a dict mapping the values to the Slang struct names.
    def get_this(self):
        return {
            "baseColor": self.b,
            "roughness": self.r,
            "metallic": self.m,
            "specular": self.s,
            "_type": "Properties",
        }


def _extract_epoch_from_name(fn: str) -> int:
    base = os.path.splitext(os.path.basename(fn))[0]
    if "_e" not in base:
        return -1
    tail = base.split("_e")[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else -1


def find_latest_model_path(preferred_path: str) -> str | None:
    if preferred_path and os.path.isfile(preferred_path):
        if preferred_path.lower().endswith(".npz"):
            return os.path.abspath(preferred_path)
        return None

    folder = (
        preferred_path
        if os.path.isdir(preferred_path)
        else (os.path.dirname(os.path.abspath(preferred_path)) or ".")
    )
    if not os.path.isdir(folder):
        return None

    npz_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".npz")
    ]
    if not npz_files:
        return None

    epoch_files = [(p, _extract_epoch_from_name(p)) for p in npz_files]
    epoch_files = [x for x in epoch_files if x[1] >= 0]
    if epoch_files:
        epoch_files.sort(key=lambda x: x[1])
        return epoch_files[-1][0]

    npz_files.sort(key=lambda p: os.path.getmtime(p))
    return npz_files[-1]


def load_model(model: nn.ModelChain, path: str):
    chosen = find_latest_model_path(path)
    if chosen is None:
        raise RuntimeError(f"No model found")

    data = np.load(chosen)
    params = model.parameters()

    for i, p in enumerate(params):
        key = f"param_{i:03d}"
        if key not in data:
            raise RuntimeError(f"Missing {key} in model")

        arr = data[key]
        if arr.size != int(np.prod(p.shape)):
            raise RuntimeError(
                f"Shape mismatch for {key}: model file has {arr.size}, "
                f"model expects {np.prod(p.shape)}"
            )

        p.copy_from_numpy(arr)

    print(f"[OK] Loaded {len(params)} parameters from {chosen}")
    return chosen


def choose_model_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[ERROR] tkinter not available")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select model (.npz)",
        filetypes=[("Neural BRDF model", "*.npz")],
        initialdir=os.getcwd(),
    )

    root.destroy()
    return path if path else None


def build_ui(ctx: spy.ui.Context):
    if not hasattr(build_ui, "initialized"):
        build_ui.initialized = False
        build_ui.win = None
        build_ui.bar = None

    if not build_ui.initialized:
        build_ui.bar = spy.ui.Window(
            ctx.screen,
            title="Panels",
            position=spy.float2(0, 0),
            size=spy.float2(resolution, 75),
        )
        spy.ui.Text(
            build_ui.bar,
            "Neural".rjust(72),
        )

        # --- Main UI window ---
        build_ui.win = spy.ui.Window(
            ctx.screen,
            title="Neural BRDF (Inference)",
            position=spy.float2(0, 700),
            size=spy.float2(resolution, 350),
        )

        def do_load():
            global save_status, save_status_until, paused

            paused = True

            path = choose_model_file()
            if not path:
                save_status = "Load cancelled."
                save_status_until = time.time() + 2.0
                return

            try:
                load_model(model, path)
                save_status = f"Loaded model: {path}"
            except Exception as e:
                save_status = f"Load failed: {e}"

            save_status_until = time.time() + 2.0

        spy.ui.Button(build_ui.win, "Load model", do_load)
        spy.ui.SliderFloat(
            build_ui.win,
            "roughness",
            properties.r,
            lambda v: setattr(properties, "r", v),
            0.3,
            1.0,
        )

        spy.ui.SliderFloat(
            build_ui.win,
            "metallic",
            properties.m,
            lambda v: setattr(properties, "m", v),
            0.0,
            1.0,
        )

        spy.ui.SliderFloat(
            build_ui.win,
            "specular",
            properties.s,
            lambda v: setattr(properties, "s", v),
            0.0,
            1.0,
        )

        build_ui.initialized = True

    # Show every frame
    build_ui.bar.show()
    build_ui.win.show()


# Create app and load the brdf shader.

device = spy.create_device(
    spy.DeviceType.vulkan, False, include_paths=nn.slang_include_paths()
)

app = App(device, "Neural BRDF (Inference)", width=resolution, height=resolution + 300)
device = app.device

if spy.Feature.cooperative_vector in device.features:
    print("Cooperative vector enabled!")
    mlp_input = nn.ArrayKind.coopvec
    mlp_precison = nn.Real.half
else:
    print(
        "Device does not support cooperative vector. Sample will run, but it will be slowe"
    )
    mlp_input = nn.ArrayKind.array
    mlp_precison = nn.Real.float


mode_tag = "coop" if mlp_input == nn.ArrayKind.coopvec else "float"
model_path = f"neural_brdf_{mode_tag}.npz"

# Setup the camera with the app's frame dimensions.
camera = Camera(app)

# BRDF lighting parameters.
properties = Properties(spy.float3(0.82, 0.2, 0.6), 0.4, 0.7, 0.5)


# Neural network model for the BRDF
model = nn.ModelChain(
    nn.Convert.to_array(),
    nn.FrequencyEncoding(6),
    nn.Convert.to_precision(mlp_precison),
    nn.Convert.to_array_kind(mlp_input),
    nn.LinearLayer(nn.Auto, 32),
    nn.LeakyReLU(),
    nn.LinearLayer(nn.Auto, 32),
    nn.LeakyReLU(),
    nn.LinearLayer(nn.Auto, 32),
    nn.LeakyReLU(),
    nn.LinearLayer(nn.Auto, 4),
    nn.Convert.to_vector(),
    nn.Convert.to_precision(nn.Real.float),
    nn.Exp(),
)

module = Module.load_from_file(device, "brdf.slang")
model.initialize(module, "float[5]")
try:
    chosen = load_model(model, model_path)
    print(f"Using: {chosen}")
except Exception as e:
    print(f"No model loaded: {e}")

timer = Timer()


app.build_ui = build_ui


inference_output = spy.Tensor.empty(
    device, shape=(resolution, resolution), dtype=spy.float4
)

last_update = 0.0

# Run the app.
while app.process_events():

    timer.start()

    command_encoder = device.create_command_encoder()

    id = device.submit_command_buffer(command_encoder.finish())
    # Stall and wait
    device.wait_for_submit(id)

    module.computeBRDFNeural(
        model, camera, properties, call_id(), _result=inference_output
    )
    app.blit(
        inference_output,
        size=spy.int2(resolution, resolution),
        tonemap=False,
        bilinear=False,
    )

    app.present()

    timer.stop()

    now = time.time()
    if now - last_update > 0.25:
        dt = timer.elapsed()
        fps = 0.0 if dt <= 0 else 1.0 / dt

        app.window.title = f"Neural BRDF | {fps:6.1f} FPS"

        last_update = now
