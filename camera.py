import numpy as np
import pyrr

class Camera:
    """
    Handles 3D camera logic for the viewer, including orbit, pan, and zoom controls.

    This class maintains the camera's state (position, target, orientation) and provides
    methods to calculate the view matrix and handle user input events from `wgpu`.

    Attributes:
        canvas (WgpuCanvas): The canvas instance to request redraws from.
        target (np.ndarray): The 3D point the camera is looking at. Shape (3,).
        distance (float): Distance of the camera eye from the target.
        azimuth (float): Horizontal orbital rotation in radians.
        elevation (float): Vertical orbital rotation in radians.
        position (np.ndarray): Current world position of the camera eye. Shape (3,).
        _dragging_left (bool): State flag for left mouse button drag.
        _dragging_right (bool): State flag for right mouse button drag.
        _input_state (dict): Stores previous mouse position for delta calculation.
    """

    def __init__(self, canvas=None):
        """
        Initialize the Camera with default viewing parameters.

        Args:
            canvas (WgpuCanvas, optional): Canvas for triggering redraws. Defaults to None.
        """
        self.canvas = canvas
        
        # State
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 5.0
        self.azimuth = 0.0   # Horizontal rotation
        self.elevation = 0.5 # Vertical rotation (radians)
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        
        # Interaction state
        self._dragging_left = False
        self._dragging_right = False
        self._input_state = {"last_pos": None}
        
    def get_view_matrix(self):
        """
        Calculate and return the 4x4 view matrix based on current spherical coordinates.

        The camera position is calculated using spherical coordinates (r, theta, phi)
        centered at `self.target`. The up vector is fixed to +Z.

        Returns:
            np.ndarray: A 4x4 flattened view matrix (column-major order expected by WebGPU).
        """
        # Calculate eye position from spherical coordinates (Z-up)
        # x = r * cos(el) * cos(az)
        # y = r * cos(el) * sin(az)
        # z = r * sin(el)
        x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.distance * np.sin(self.elevation)
        
        eye = self.target + np.array([x, y, z], dtype=np.float32)
        self.position = eye
        
        # Create look_at matrix with Z-up
        return pyrr.matrix44.create_look_at(
            eye=eye,
            target=self.target,
            up=[0.0, 0.0, 1.0]
        )

    def handle_event(self, event):
        """
        Process user input events to update camera state.

        Handles:
            - 'pointer_down': Start dragging (left=orbit, right=pan).
            - 'pointer_up': Stop dragging.
            - 'pointer_move': Update azimuth/elevation (orbit) or target (pan).
            - 'wheel': Adjust distance (zoom).

        Args:
            event (dict): Event dictionary containing 'event_type', 'x', 'y', 'button', 'dy'.
        """
        event_type = event["event_type"]
        
        if event_type == "pointer_down":
            if event["button"] == 1:
                self._dragging_left = True
                self._input_state["last_pos"] = (event["x"], event["y"])
            elif event["button"] == 2:
                self._dragging_right = True
                self._input_state["last_pos"] = (event["x"], event["y"])
                
        elif event_type == "pointer_up":
            if event["button"] == 1:
                self._dragging_left = False
            elif event["button"] == 2:
                self._dragging_right = False
            
            # Reset if no buttons are held
            if not self._dragging_left and not self._dragging_right:
                self._input_state["last_pos"] = None
                
        elif event_type == "pointer_move":
            if (self._dragging_left or self._dragging_right) and self._input_state["last_pos"]:
                curr_x, curr_y = event["x"], event["y"]
                last_x, last_y = self._input_state["last_pos"]
                
                dx = curr_x - last_x
                dy = curr_y - last_y
                
                if self._dragging_left:
                    # Orbit
                    factor = 0.01
                    self.azimuth -= dx * factor
                    self.elevation += dy * factor
                    limit = np.pi / 2 - 0.01
                    self.elevation = np.clip(self.elevation, -limit, limit)
                    
                elif self._dragging_right:
                    # Pan
                    # Calculate view vectors (Z-up)
                    x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
                    y = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
                    z = self.distance * np.sin(self.elevation)
                    eye = self.target + np.array([x, y, z], dtype=np.float32)
                    
                    forward = pyrr.vector.normalize(self.target - eye)
                    right = pyrr.vector.normalize(np.cross(forward, [0.0, 0.0, 1.0]))
                    up = pyrr.vector.normalize(np.cross(right, forward))
                    
                    # Pan factor depends on distance
                    factor = 0.002 * self.distance
                    
                    self.target -= (right * dx * factor)
                    self.target += (up * dy * factor)

                self._input_state["last_pos"] = (curr_x, curr_y)
                
                if self.canvas:
                    self.canvas.request_draw()
                    
        elif event_type == "wheel":
            # Zoom
            delta = event["dy"]
            zoom_factor = 0.001 * self.distance
            self.distance += delta * zoom_factor
            
            if self.distance < 0.1:
                self.distance = 0.1
                
            if self.canvas:
                self.canvas.request_draw()
