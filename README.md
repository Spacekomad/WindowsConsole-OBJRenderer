# WindowsConsole-OBJRenderer

Render 3D OBJ model files directly in the Windows Console with color output.  

🎥 **Demo Video**  
[Watch on YouTube](https://youtu.be/vxZlHLDut2k)

---

## What We Made

### Console Monitor Class
- Handles output of rendered images in the Windows console.  
- Since the Windows console supports only a limited set of colors, this class introduces a **3x3 RGB pixel system** to represent a wider variety of colors.  

### Model Class
- Loads OBJ-format files and stores data about faces, vertices, and textures.  
- Quadrilateral faces are divided into triangles **without generating new vertices**.  
- Faces with more than 4 vertices are not supported (assumed not to be present).  
- Assumes the use of a **single texture**.  

### Scene Class
- Stores objects to render, lights, the main directional light, and the sky color.  

### GameObject Class
- Stores each renderable object’s model data along with its **position, rotation, and scale**.  

### Vertex Shader
- Transforms each vertex into **clip space coordinates** using MVP transformation.  

### Rasterizer
- Performs perspective division, back-face culling, viewport transformation, scan conversion, and clipping.  
- Clipping does not generate new vertices—faces outside the viewport simply do not produce fragments.  
- During scan conversion, fills the depth buffer and applies **Z-test**.  

### Fragment Shader
- Determines the final pixel color.  
- Currently, only diffuse texture colors are used (lighting not implemented).  

---

## Limitations
- Vertex and fragment shaders are not programmable—only predefined operations are supported.  
- Lighting calculations are not yet implemented, so only diffuse textures are displayed.  

---

## 📁 Project Structure

```
WindowsConsole-OBJRenderer
├── Models/ # Models and textures used in the demo video
├── OBJRenderer/ # Source code
```

