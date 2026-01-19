import numpy as np
import pyrr

class Camera:
    def __init__(self, canvas=None):
        self.canvas = canvas
        
        # State
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 5.0
        self.azimuth = 0.0   # Horizontal rotation
        self.elevation = 0.5 # Vertical rotation (radians)
        
        # Interaction state
        self._dragging_left = False
        self._dragging_right = False
        self._input_state = {"last_pos": None}
        
    def get_view_matrix(self):
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
        """Handle wgpu gui events."""
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
