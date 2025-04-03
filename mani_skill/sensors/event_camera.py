import time
import torch
import sapien.core as sapien

from .camera import Camera
from .camera import EventCameraConfig  
from typing import Dict

from dataclasses import dataclass
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from .camera import CameraConfig, ShaderConfig

class EventCamera(Camera):
    """
    A subclass of the ManiSkill3 Camera that simulates an event camera.
    It compares consecutive frames to produce events in the form (x, y, t, polarity),
    then builds an event map where each pixel is:
       +1 for a positive event,
       -1 for a negative event,
        0 for no event.
    The event map is then flattened to shape (1, H*W) to pass to the environment.
    """
    def __init__(self, camera_config: EventCameraConfig, scene, articulation=None):
        super().__init__(camera_config, scene, articulation)
        
        self.event_threshold = getattr(camera_config, 'event_threshold', 0.2)
        self.use_log_intensity = getattr(camera_config, 'use_log_intensity', False)
        
        self._previous_frame = None
        self._previous_time = None
        
        # A list to store events: each event is (x, y, time, polarity)
        self._current_events = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def capture(self):
        """
        Capture a new frame from the SAPIEN camera and compute events by comparing with the previous frame.
        """
        self.camera.take_picture() # Takes current sim picture
        rgb_tex = self.camera.get_picture(["Color"]) 
        if len(rgb_tex) == 0:
            self._current_events = []
            return
        
        rgb_img = rgb_tex[0]  # shape: (H, W, 4) RGBA in [0,255]
        # Convert to grayscale in [0,1]
        gray_img = self._to_grayscale(rgb_img[..., :3])
        if self.use_log_intensity:
            gray_img = torch.log(gray_img + 1e-5)
        
        if self._previous_frame is not None:
            self._current_events = self._generate_events(gray_img, self._previous_frame)
        else:
            self._current_events = []
        
        self._previous_frame = gray_img

    def _to_grayscale(self, rgb_img: torch.Tensor) -> torch.Tensor:
        """
        Convert an (H, W, 3) RGB image (in [0,255]) to a grayscale image in [0,1].
        """
        if rgb_img.dtype == torch.uint8:
            rgb_img = rgb_img.float() / 255.0
        return 0.299 * rgb_img[..., 0] + 0.587 * rgb_img[..., 1] + 0.114 * rgb_img[..., 2]
        # return torch.dot(rgb_img, torch.as_tensor([0.299, 0.587, 0.114], device=self.device)) / 255.0

    def _generate_events(
        self,
        current_frame: torch.Tensor,
        prev_frame: torch.Tensor,
    ):
        """
        Compute pixel differences and return a list of events.
        Each event is (x, y, time, polarity), with polarity +1 if intensity increased,
        and -1 if it decreased. More events == Longer list. This is the inconsistent bandwidth issue!
        """
        dI = current_frame - prev_frame
        event_mask = torch.abs(dI) > self.event_threshold
        if not torch.any(event_mask):
            return []
        
        polarity_values = torch.where(dI[event_mask] > 0, 1, -1)
        batch, ys, xs = torch.where(event_mask)  # both of shape (N,)
        
        events = []
        for i, (y, x) in enumerate(zip(ys, xs)):
            events.append((x, y, polarity_values[i]))
        return events

    def get_event_frame(self) -> torch.Tensor:
        """
        Build a 2D event map of shape (H, W) where each pixel is:
            +1 if a positive event occurred,
            -1 if a negative event occurred,
             0 otherwise.
        Then flatten the event map to shape (1, H*W).
        """
        H, W = self.camera.height, self.camera.width
        event_map = torch.zeros((H, W), dtype=torch.int8, device=self.device)
        for (x, y, t, pol) in self._current_events:
            x_i = int(x)
            y_i = int(y)
            if 0 <= x_i < W and 0 <= y_i < H:
                event_map[y_i, x_i] = pol
        return event_map.reshape(1, -1)

    def get_obs(self,
                rgb: bool = True,
                depth: bool = True,
                position: bool = True,
                segmentation: bool = True,
                normal: bool = False,
                albedo: bool = False,
                apply_texture_transforms: bool = True):
        """
        Return a dictionary with a single key "rgb" that contains the flattened event frame.
        The flattened frame is a 2D array of shape (1, H*W) with values -1, 0, +1.
        """
        images_dict = {}
        
        # Capture a new frame and compute events
        self.capture()
        flattened_event_frame = self.get_event_frame()
        images_dict['event_camera'] = flattened_event_frame
        
        return images_dict