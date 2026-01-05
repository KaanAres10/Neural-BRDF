# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional
import slangpy as spy


class App:
    def __init__(
        self,
        device: spy.Device,
        title: str = "BRDF Example",
        width: int = 1024,
        height: int = 1024,
    ):
        super().__init__()

        # Create a window
        self._window = spy.Window(width=width, height=height, title=title, resizable=True)

        # Create a device with local include path for shaders
        self._device = device

        self._module = spy.Module.load_from_file(self._device, "app.slang")

        # Setup swapchain
        self.surface = self._device.create_surface(self._window)
        self.surface.configure(width=self._window.width, height=self._window.height)

        # Will contain output texture
        self._output_texture: spy.Texture = self.device.create_texture(
            format=spy.Format.rgba16_float,
            width=width,
            height=height,
            mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="output_texture",
        )

        # Store mouse pos
        self._mouse_pos = spy.float2()

        # Internal events
        self._window.on_keyboard_event = self._on_window_keyboard_event
        self._window.on_mouse_event = self._on_window_mouse_event
        self._window.on_resize = self._on_window_resize

        # Hookable events
        self.on_keyboard_event: Optional[Callable[[spy.KeyboardEvent], None]] = None
        self.on_mouse_event: Optional[Callable[[spy.MouseEvent], None]] = None

        self._ui = spy.ui.Context(self._device)  
        self._ui_wants_mouse = False
        self._ui_wants_keyboard = False
        self.build_ui = None

    @property
    def device(self) -> spy.Device:
        return self._device

    @property
    def window(self) -> spy.Window:
        return self._window

    @property
    def mouse_pos(self) -> spy.float2:
        return self._mouse_pos

    @property
    def output(self) -> spy.Texture:
        return self._output_texture

    def process_events(self):
        if self._window.should_close():
            return False
        self._window.process_events()
        return True

    def present(self):
        if not self.surface.config:
            return
        image = self.surface.acquire_next_image()
        if not image:
            return

        if (
            self._output_texture == None
            or self._output_texture.width != image.width
            or self._output_texture.height != image.height
        ):
            self._output_texture = self.device.create_texture(
                format=spy.Format.rgba16_float,
                width=image.width,
                height=image.height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="output_texture",
            )

        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(image, self._output_texture)

        if self._ui:
            # Begin UI frame (most builds use width/height; yours might ignore args, but this works if accepted)
            try:
                self._ui.begin_frame(image.width, image.height)
            except TypeError:
                self._ui.begin_frame()

            # Build UI widgets
            if getattr(self, "build_ui", None):
                self.build_ui(self._ui)

            # End UI frame + render into the swapchain image
            try:
                # most common
                self._ui.end_frame(image, command_encoder)
            except TypeError:
                # sometimes swapped
                self._ui.end_frame(command_encoder, image)

        command_encoder.set_texture_state(image, spy.ResourceState.present)
        self._device.submit_command_buffer(command_encoder.finish())

        del image
        self.surface.present()

    def blit(
        self,
        source: spy.Tensor,
        size: Optional[spy.int2] = None,
        offset: Optional[spy.int2] = None,
        tonemap: bool = True,
        bilinear: bool = False,
    ):
        if len(source.shape) != 2:
            raise ValueError("Source tensor must be 2D (height, width).")
        if size is None:
            size = spy.int2(source.shape[1], source.shape[0])
        if offset is None:
            offset = spy.int2(0, 0)
        self._module.blit(
            spy.grid((size.y, size.x)),
            size,
            offset,
            tonemap,
            bilinear,
            source,
            self.output,
        )

    def _on_window_keyboard_event(self, event: spy.KeyboardEvent):
        if self._ui and self._ui.handle_keyboard_event(event):  
            return

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self._window.close()
                return
            elif event.key == spy.KeyCode.f1:
                if self._output_texture:
                    spy.tev.show_async(self._output_texture)
                return
            elif event.key == spy.KeyCode.f2:
                if self._output_texture:
                    bitmap = self._output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")
                return
        if self.on_keyboard_event:
            self.on_keyboard_event(event)

    def _on_window_mouse_event(self, event: spy.MouseEvent):
        if self._ui and self._ui.handle_mouse_event(event):    
            return
        if event.type == spy.MouseEventType.move:
            self._mouse_pos = event.pos
        if self.on_mouse_event:
            self.on_mouse_event(event)

    def _on_window_resize(self, width: int, height: int):
        self._device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height)
        else:
            self.surface.unconfigure()
