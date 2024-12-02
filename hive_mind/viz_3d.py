import math

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    NodePath, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, Point3, DirectionalLight,
    AmbientLight, Vec4, WindowProperties, TextNode
)
from typing import Optional, Tuple, List
import numpy as np


class AnimatedHillPlotter(ShowBase):
    def __init__(self, window_size: Tuple[int, int] = (1024, 768)):
        """
        Initialize the Panda3D-based hill plotter.
        
        Args:
            window_size: Tuple of (width, height) for the window
        """
        ShowBase.__init__(self)
        
        # Set window properties
        props = WindowProperties()
        props.setSize(window_size[0], window_size[1])
        self.win.requestProperties(props)
        
        # Setup scene
        self._setup_scene()
        
        # Initialize state variables
        self.current_mesh: Optional[NodePath] = None
        self.peak_markers: List[NodePath] = []
        self.current_heights: Optional[np.ndarray] = None
        self.target_heights: Optional[np.ndarray] = None
        self.interpolation_steps: int = 0
        self.current_step: int = 0
        self.mesh_shape: Optional[Tuple[int, int]] = None
        self.continue_flag: bool = False
        
        # Setup key handling
        self.accept('n', self.handle_next_key)

    def create_sphere(self, radius: float = 0.2, segments: int = 16) -> NodePath:
        """
        Create a sphere geometry for peak markers.
        
        Args:
            radius: Radius of the sphere
            segments: Number of segments (higher = smoother sphere)
        
        Returns:
            NodePath containing the sphere geometry
        """
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('sphere', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')

        # Create vertices
        for lat in range(segments):
            lat_rad = math.pi * float(lat) / segments
            for lon in range(segments):
                lon_rad = 2.0 * math.pi * float(lon) / segments
                
                x = radius * math.sin(lat_rad) * math.cos(lon_rad)
                y = radius * math.sin(lat_rad) * math.sin(lon_rad)
                z = radius * math.cos(lat_rad)
                
                vertex.addData3(x, y, z)
                normal.addData3(x/radius, y/radius, z/radius)
                color.addData4(1, 0, 0, 1)  # Red color

        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        for lat in range(segments - 1):
            for lon in range(segments - 1):
                v1 = lat * segments + lon
                v2 = v1 + 1
                v3 = v1 + segments
                v4 = v3 + 1
                
                tris.addVertices(v1, v2, v3)
                tris.addVertices(v2, v4, v3)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        node = GeomNode('sphere')
        node.addGeom(geom)
        return self.render.attachNewNode(node)

    def _setup_scene(self):
        """Set up the 3D scene with lighting and camera."""
        # Setup camera
        self.camera.setPos(0, -20, 10)
        self.camera.lookAt(0, 0, 0)
        
        # Add lights
        dlight = DirectionalLight('dlight')
        dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(45, -45, 0)
        self.render.setLight(dlnp)
        
        # Add ambient light
        alight = AmbientLight('alight')
        alight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
    
    def create_terrain_mesh(self, heights: np.ndarray) -> NodePath:
        """
        Create a terrain mesh from height data.
        
        Args:
            heights: 2D numpy array of height values
        """
        rows, cols = heights.shape
        
        # Create vertex data format
        format = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData('terrain', format, Geom.UHStatic)
        
        # Create vertex writers
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        
        # Add vertices
        for y in range(rows):
            for x in range(cols):
                # Vertex position
                vertex.addData3(x - cols/2, y - rows/2, heights[y, x])
                
                # Simple normal calculation (pointing up)
                normal.addData3(0, 0, 1)
                
                # Color based on height (creating a height-based gradient)
                height_color = heights[y, x] / np.max(heights)
                color.addData4(height_color, height_color * 0.5, 1.0, 1.0)
        
        # Create triangles
        tris = GeomTriangles(Geom.UHStatic)
        for y in range(rows - 1):
            for x in range(cols - 1):
                # Add two triangles for each grid cell
                v1 = y * cols + x
                v2 = v1 + 1
                v3 = v1 + cols
                v4 = v3 + 1
                
                tris.addVertices(v1, v2, v3)
                tris.addVertices(v2, v4, v3)
        
        # Create the geometry
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        
        # Create and return the node
        node = GeomNode('terrain')
        node.addGeom(geom)
        return self.render.attachNewNode(node)
    
    def update_plot(self, 
                   hill_image: np.ndarray, 
                   title: str, 
                   peaks: Optional[List[Tuple[int, int]]] = None,
                   smooth_factor: float = 0.1,
                   wait_for_key: bool = True) -> None:
        """
        Update the plot with new height data.
        
        Args:
            hill_image: 2D numpy array of height values
            title: Title for the plot (currently unused in Panda3D version)
            peaks: List of peak coordinates (x, y)
            smooth_factor: Factor controlling transition smoothness (0-1)
            wait_for_key: If True, wait for 'n' key press to continue
        """
        if self.current_mesh is None:
            self.current_mesh = self.create_terrain_mesh(hill_image)
            self.current_heights = hill_image
            self.mesh_shape = hill_image.shape
        else:
            # Setup interpolation
            self.target_heights = hill_image
            self.interpolation_steps = int(1 / smooth_factor)
            self.current_step = 0
            
            # Start interpolation task
            self.doMethodLater(0.05, self.interpolate_heights, 'interpolate_heights')
        
        # Update peak markers
        for marker in self.peak_markers:
            marker.removeNode()
        self.peak_markers.clear()
        
        if peaks:
            for peak_x, peak_y in peaks:
                peak_x_int = int(np.clip(peak_x, 0, hill_image.shape[1] - 1))
                peak_y_int = int(np.clip(peak_y, 0, hill_image.shape[0] - 1))
                peak_z = hill_image[peak_y_int, peak_x_int]

                # Create peak marker using our sphere creation method
                marker = self.create_sphere(radius=0.2)
                marker.setPos(peak_x - hill_image.shape[1]/2, 
                            peak_y - hill_image.shape[0]/2, 
                            peak_z + 0.5)
                self.peak_markers.append(marker)

    def interpolate_heights(self, task):
        """Task to handle height interpolation animation."""
        if self.current_heights is None or self.target_heights is None:
            return Task.done
        
        alpha = self.current_step / float(self.interpolation_steps - 1)
        interpolated = ((1 - alpha) * self.current_heights + 
                       alpha * self.target_heights)
        
        # Update mesh
        self.current_mesh.removeNode()
        self.current_mesh = self.create_terrain_mesh(interpolated)
        
        self.current_step += 1
        
        if self.current_step >= self.interpolation_steps:
            self.current_heights = self.target_heights
            return Task.done
        
        return Task.again
    
    def handle_next_key(self):
        """Handle 'n' key press."""
        self.continue_flag = True
    
    def initialize_window(self, initial_shape: Tuple[int, int]) -> None:
        """Initialize with initial shape."""
        self.mesh_shape = initial_shape
        self.current_heights = np.zeros(initial_shape)
        self.current_mesh = self.create_terrain_mesh(self.current_heights)
    
    def close(self) -> None:
        """Close the window."""
        self.userExit()
