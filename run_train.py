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


save_status = ""  # text shown in UI
save_status_until = 0.0  # time.time() deadline


def save_model(model: nn.ModelChain, path: str, epoch: int | None = None):
    if epoch is not None:
        root, ext = os.path.splitext(path)
        path = f"{root}_e{epoch:06d}{ext}"

    params = {}
    for i, p in enumerate(model.parameters()):
        params[f"param_{i:03d}"] = p.to_numpy()

    np.savez(path, **params)
    print(f"[OK] Saved {len(params)} parameters to {path}")


def load_model(model: nn.ModelChain, path: str):
    if os.path.isdir(path):
        files = []
        for fn in os.listdir(path):
            if fn.lower().endswith(".npz"):
                files.append(fn)

        if not files:
            raise RuntimeError(f"No .npz checkpoints found in folder: {path}")

        def extract_epoch(fn: str) -> int:
            try:
                base = os.path.splitext(fn)[0]
                if "_e" in base:
                    return int(base.split("_e")[-1])
            except Exception:
                pass
            return -1

        files.sort(key=extract_epoch)
        path = os.path.join(path, files[-1])

    data = np.load(path)
    params = model.parameters()

    for i, p in enumerate(params):
        key = f"param_{i:03d}"
        if key not in data:
            raise RuntimeError(f"Missing {key} in model")

        arr = data[key]
        if arr.size != int(np.prod(p.shape)):
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"model file has {arr.size}, model expects {np.prod(p.shape)}"
            )

        p.copy_from_numpy(arr)

    print(f"[OK] Loaded {len(params)} parameters from {path}")
    return path


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
            size=spy.float2(resolution * 3 + 10 * 2, 75),
        )
        spy.ui.Text(
            build_ui.bar,
            "Reference".rjust(70) + "Neural".rjust(145) + "Difference".rjust(150),
        )

        # --- Main UI window ---
        build_ui.win = spy.ui.Window(
            ctx.screen,
            title="Neural BRDF (Training)",
            position=spy.float2(0, 700),
            size=spy.float2(resolution * 3 + 10 * 2, 350),
        )

        spy.ui.CheckBox(
            build_ui.win,
            "Pause training",
            paused,
            lambda v: globals().__setitem__("paused", v),
        )

        def do_save():
            global save_status, save_status_until
            save_model(model, model_path, epoch)
            save_status = f"Saved: {model_path}"
            save_status_until = time.time() + 2.0

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

                build_optimizer()

                save_status = f"Loaded model: {path} (training paused)"
            except Exception as e:
                save_status = f"Load failed: {e}"

            save_status_until = time.time() + 2.0

        spy.ui.Button(build_ui.win, "Save model", do_save)
        spy.ui.Button(build_ui.win, "Load model", do_load)
        spy.ui.SliderFloat(
            build_ui.win,
            "roughness",
            properties.r,
            lambda v: setattr(properties, "r", v),
            0.0,
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
    spy.DeviceType.vulkan,
    False,
    include_paths=nn.slang_include_paths(),
    enable_print=True,
)

app = App(
    device,
    "Neural BRDF (Training)",
    width=resolution * 3 + 10 * 2,
    height=resolution + 300,
)
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


def build_optimizer():
    global optim, grad_scale
    optim = nn.AdamOptimizer(learning_rate=0.001)

    grad_scale = 1.0
    if mlp_precison == nn.Real.half:
        optim = nn.FullPrecisionOptimizer(optim, gradient_scale=128.0)
        grad_scale = 128.0

    optim.initialize(module, model.parameters())


# Create optimizer
build_optimizer()


batch_shape = (256, 256)
loss_scale = grad_scale / (math.prod(batch_shape) * 4.0)
num_batches_per_epoch = 10

# Initalize a sepearte random number generator per batch entry
seeds = np.random.get_bit_generator().random_raw(batch_shape).astype(np.uint32)
rng = module.RNG(seeds)

timer = Timer()


app.build_ui = build_ui


reference_output = spy.Tensor.empty(
    device, shape=(resolution, resolution), dtype=spy.float4
)
training_output = spy.Tensor.empty_like(reference_output)
difference_output = spy.Tensor.empty_like(reference_output)
loss_buffer = NDBuffer(device, "float", shape=(batch_shape[0] * batch_shape[1],))

print_every = 100
epoch = 0
paused = False

last_update = 0.0

# Run the app.
while app.process_events():

    timer.start()

    command_encoder = device.create_command_encoder()

    if not paused:
        for i in range(num_batches_per_epoch):
            module.trainBRDF.append_to(
                command_encoder,
                model,
                rng,
                loss_scale,
                call_id(),
                spy.int2(batch_shape[0], batch_shape[1]),
                loss_buffer,
            )
        optim.step(command_encoder)

    id = device.submit_command_buffer(command_encoder.finish())
    # Stall and wait
    device.wait_for_submit(id)

    device.flush_print()

    loss_np = loss_buffer.to_numpy()

    finite = loss_np[np.isfinite(loss_np)]
    mean_loss = float(np.mean(finite)) if finite.size > 0 else float("nan")

    # if epoch % print_every == 0:
    #     print(f"[epoch {epoch:05d}] mean loss = {mean_loss:.6e}")

    epoch += 1

    msamples = (num_batches_per_epoch * math.prod(batch_shape)) * 1e-6
    # print(
    #     f"Throughput: {timer.frequency() * msamples:.2f} MSamples/s "
    #     f"Epoch time: {timer.elapsed() * 1e3:.1f}ms"
    # )

    offset = 0
    module.computeBRDF(camera, properties, call_id(), _result=reference_output)
    app.blit(
        reference_output,
        size=spy.int2(resolution, resolution),
        offset=spy.int2(offset, 0),
        tonemap=False,
        bilinear=False,
    )

    offset += resolution + 10

    module.computeBRDFNeural(
        model, camera, properties, call_id(), _result=training_output
    )
    app.blit(
        training_output,
        size=spy.int2(resolution, resolution),
        offset=spy.int2(offset, 0),
        tonemap=False,
        bilinear=False,
    )
    offset += resolution + 10

    module.absDiff(reference_output, training_output, _result=difference_output)
    app.blit(
        difference_output,
        size=spy.int2(resolution, resolution),
        offset=spy.int2(offset, 0),
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
