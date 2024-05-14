# Blender addon to 
# 1. enable user to draw a besier curve using the mouse
# 2. sample the curve uniformly
# 3. select segments of the curve and assign a label to it
# 4. run the motion graph algorithm on the curve segments to find optimal motions
# 5. get final animation. 

# import bpy
# from graphv2 import TransGraph, MoGraph
# from utils import *
# import options as opt

import bpy
import mathutils
import numpy as np
from bpy.props import ( IntProperty , PointerProperty)
from bpy.types import ( PropertyGroup )

class MG_Properties(PropertyGroup):
    N: IntProperty(name="Num of Samples [N]", default=60, min=15, max=120)
    P: IntProperty(name="Num of Segments [P]", default=1, min=2, max=15)
    Label: bpy.props.EnumProperty(
                name="Segment Label",
                items=[
                    ('0', "walk", "Label as 0"),
                    ('1', "run", "Label as 1"),
                    ('2', "violin", "Label as 2"),
                    ('3', "basket ball", "Label as 3"),
                ]
            )

class FindSegments(bpy.types.Operator):
    bl_label = "Label Segments"
    bl_idname = "object.label_segments"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        curve = context.object.data.splines.active
        num_segments = len(curve.bezier_points) - 1

        # assign P to the number of segments
        context.scene.mograph_tools.P = num_segments
        return {'FINISHED'}

class DrawBezierCurveOperator(bpy.types.Operator):
    bl_idname = "object.draw_curve"
    bl_label = "Draw Bezier Curve"
    bl_options = {"REGISTER", "UNDO"}
    
    def drawBounding(self, context):
        # create a plane in z axis to bound the curve
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    
    def execute(self, context):
        # select all objects and delete
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)
        # create a plane in z axis to bound the curve
        self.drawBounding(context)
        # create a warning dialog to the user to draw the curve in the bounding box
        self.report({'WARNING'}, "Draw the curve in the bounding box")
        # add a bezier curve
        bpy.ops.curve.primitive_bezier_curve_add()
        bpy.ops.object.editmode_toggle()
        bpy.ops.curve.delete(type='VERT')
        bpy.ops.wm.tool_set_by_id(name="builtin.brush")
        return {'FINISHED'}  

class SampleBezierCurveOperator(bpy.types.Operator):
    bl_idname = "object.sample_curve"
    bl_label = "Sample the Bezier Curve"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in active object is mesh
            if bpy.context.object and bpy.context.object.type == 'CURVE':
                return True
            else:
                return False
        except: return False

    def sample_bezier_curve_uniformly(self, curve, num_samples):
        # sample each bezier segement into 60 points     

        # get the number of control points of bezier curve
        num_control_points = len(curve.bezier_points)

        # get the number of segments of bezier curve
        num_segments = num_control_points - 1
        total_points = []
        for i in range(num_segments):
            if i == 0:
                resolution = num_samples
            else:
                resolution = num_samples + 1
            knot1 = curve.bezier_points[i].co
            knot2 = curve.bezier_points[i+1].co
            handle1 = curve.bezier_points[i].handle_right
            handle2 = curve.bezier_points[i+1].handle_left
            points = mathutils.geometry.interpolate_bezier(knot1, handle1, handle2, knot2, resolution)
            
            # lets remove duplicate points
            if i > 0:
                total_points.extend(points[1:])
            else:
                total_points.extend(points)
        return total_points

    def create_mesh_from_samples(self, samples):
        print("samples saved")
        np.save("/Users/meghanarao/Documents/github(thesis)/paths/new.npy", samples)
        mesh = bpy.data.meshes.new(name="MyMesh")
        mesh_object = bpy.data.objects.new("MyMeshObject", mesh)
        bpy.context.collection.objects.link(mesh_object)
        bpy.context.view_layer.objects.active = mesh_object
        mesh_object.select_set(True)

        vertices = []
        edges = []
        for index, coord in enumerate(samples):
            vertices.append(coord)
            if not (index == 0):
                edges.append(tuple([index-1, index]))
                
        mesh.from_pydata(vertices, edges, [])
        mesh.update()
        bpy.ops.object.mode_set(mode='EDIT')

    def execute(self, context):
        N = context.scene.mograph_tools.N
        curve = context.object.data.splines.active
        samples = self.sample_bezier_curve_uniformly(curve, N)
        # delete the curve, in edit mode
        bpy.ops.curve.delete(type='VERT')
        self.create_mesh_from_samples(samples)

        return {'FINISHED'}  
       
        
# UI panels!
class PT_Draw_Curve(bpy.types.Panel):
    bl_label  = "Draw Bezier Curve"  
    bl_idname = "PT_Draw_Curve"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoGraph"

    def draw(self, context):
        layout = self.layout
        
        draw_curve = layout.row()
        draw_curve.operator("object.draw_curve", text="Draw Bezier Curve",icon ="STROKE")

        # find number of segments in the curve
        find_segments = layout.row()
        find_segments.operator("object.label_segments", text="Find Segments", icon = "MOD_CURVE")

class PT_Label_Segments(bpy.types.Panel):
    bl_label = "Label Segments"
    bl_idname = "PT_Label_Segments"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoGraph"

    def draw(self, context):
        # for each segment, add a drop down property with 5 options for the user to select
        layout = self.layout

        labels = []

        # for each segment choose a label from the drop down and append it to the labels array

        for i in range(context.scene.mograph_tools.P):
            # create a new drop down property
            row = layout.row()
            # row.prop(context.scene.mograph_tools, "Label")
            row.prop(context.scene.mograph_tools, "Label", text=f"Segment {i+1}")
            labels.append(context.scene.mograph_tools.Label)

            
class PT_Sample_Curve(bpy.types.Panel):
    bl_label  = "Sample Bezier Curve"  
    bl_idname = "PT_Sample_Curve"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoGraph"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(context.scene.mograph_tools, "N")

        sample_curve = layout.row()
        sample_curve.operator("object.sample_curve", text="Sample Bezier Curve", icon = "SEQ_LUMA_WAVEFORM")
    
def menu_func(self, context):
    self.layout.operator(DrawBezierCurveOperator.bl_idname)
    self.layout.operator(SampleBezierCurveOperator.bl_idname)
    self.layout.operator(FindSegments.bl_idname)
    
def register():
    bpy.utils.register_class(MG_Properties)
    bpy.utils.register_class(PT_Draw_Curve)
    bpy.utils.register_class(DrawBezierCurveOperator)
    bpy.utils.register_class(FindSegments)
    bpy.utils.register_class(PT_Label_Segments)
    bpy.utils.register_class(SampleBezierCurveOperator)
    bpy.utils.register_class(PT_Sample_Curve)
    bpy.types.VIEW3D_MT_curve_add.append(menu_func)
    
    bpy.types.Scene.mograph_tools = bpy.props.PointerProperty(type=MG_Properties)

def unregister():
    bpy.utils.unregister_class(MG_Properties)
    bpy.utils.unregister_class(PT_Draw_Curve)
    bpy.utils.unregister_class(DrawBezierCurveOperator)
    bpy.utils.unregister_class(FindSegments)
    bpy.utils.unregister_class(PT_Label_Segments)
    bpy.utils.unregister_class(SampleBezierCurveOperator)
    bpy.utils.unregister_class(PT_Sample_Curve)
    bpy.types.VIEW3D_MT_curve_add.remove(menu_func)
    

    del bpy.types.Scene.mograph_tools

    
if __name__ == "__main__":
    register()