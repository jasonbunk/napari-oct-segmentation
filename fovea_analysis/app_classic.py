# Author: Jason Bunk
import os
import re
import threading
from copy import deepcopy
import numpy as np
import napari
from magicgui import magicgui, widgets
from napari.layers import Image
from napari.components import LayerList
from napari.utils.events import Event
from threading_work import thread_work_loop_classic_detection, build_on_yielded
from my_utils import image_shape_to_2d_label_shape, build_spreadsheet_saver_function
from layer_group_alignment import (
    LayersGroup,
    toggle_viz_unaligned_to_aligned,
    toggle_viz_aligned_to_unaligned,
)
from definitions_and_config import (
    RetinaLayers,
    RetinaLayerColors,
    OCTAnnotationMode,
    machine_labels_opacity,
    user_label_opacity,
    metrics_table_rows,
    typical_image_width_pixels,
    typical_image_width_mm,
    parafovea_radius_mm,
    radius1_cal,
)
import globals

user_rpe_annos = None
user_points = None
layers_unaligned = None
layers_rpe_align = None
parafovea_indicator_lines = None
parafovea_indicator_opacity = 0

auto_worker = None
worker_alive = threading.Event() # used for multithreading safety. By default, state is False. The launched computations thread will set to true when it's ready.
just_created_user_labels = False
last_selected_layer_label = None
global_state_align2rpe = False
global_parafovea_width_pix = None
global_fovea_clicks = None # either contains 3 clicked points, or is None
global_hide_left_paraf = False,
global_hide_right_paraf = False
global_det_max_slope = 3


# This configures the buttons and stuff on the right side of the app.
# It's a magicgui decorated function which is called whenever the user interacts with a button or slider.
# The decorator specifies the type of each widget, and initial values.
# The function decides what to do based on the widget state... for example, change the paintbrush size if needed.
@magicgui(auto_call=True,
    parafovea_pix={"widget_type":"FloatText", "value":parafovea_radius_mm*typical_image_width_pixels/typical_image_width_mm, "max":typical_image_width_pixels},
    label_viz={"widget_type":"FloatSlider", "max":1.0, "value":1.0},
    autodets_viz={"widget_type":"FloatSlider", "max":1.0, "value":1.0},
    det_max_slope={"widget_type":"IntSlider", "max":10, "min":2, "value":3},
    retina_layer={"widget_type":"RadioButtons", "value":RetinaLayers.RPE},
    mode={"widget_type":"RadioButtons", "value":OCTAnnotationMode.Trace_Layers},
    metrics={"widget_type":"Table", "value":{"data":[[0,],]*len(metrics_table_rows), "index":metrics_table_rows, "columns":["value",]}})
def oct_widget(
    parafovea_pix:float,
    label_viz:float,
    autodets_viz:float,
    det_max_slope,
    retina_layer: RetinaLayers,
    mode: OCTAnnotationMode,
    align_to_RPE: bool,
    eraser: bool,
    hide_left_paraf: bool,
    hide_right_paraf: bool,
    metrics: dict
    ):
    """This docstring to provides helpful popup text when the mouse hovers over something in the sidebar widget.

    Parameters
    ----------
    parafovea_pix : float
        Radius in pixels around the fovea center where parafovea thickness is to be measured.
    label_viz : float
        Show or hide markup annotations; 0 = hide, 1 = show
    autodets_viz : float
        Show or hide machine-detected annotations; 0 = hide, 1 = show
    retina_layer : Enum
        Which layer to trace/correct? Note: ILM or RPE cannot be annotated in aligned-to-RPE mode.
    mode : Enum
        What should mouse click do? Either trace layer corrections; or click fovea points for fovea angle measurement.
    align_to_RPE : bool
        Click to toggle whether image and annotations should be aligned to the RPE layer.
    eraser : bool
        Click to use the mouse to erase your annotations.
    metrics : Table
        Table of measurements. Won't be valid without clicking 3 fovea points. Distance unit: millimeters; angle unit: degrees.
    """
    global user_rpe_annos, user_points, layers_unaligned, layers_rpe_align, last_selected_layer_label, global_state_align2rpe, global_parafovea_width_pix, global_fovea_clicks, parafovea_indicator_lines, parafovea_indicator_opacity, global_hide_left_paraf, global_hide_right_paraf, global_det_max_slope
    if user_rpe_annos is None or user_points is None or layers_unaligned is None or layers_rpe_align is None or align_to_RPE is None:
        return
    do_update_annotations = (parafovea_pix != global_parafovea_width_pix or global_hide_left_paraf != hide_left_paraf or global_hide_right_paraf != hide_right_paraf or det_max_slope != global_det_max_slope)
    global_hide_left_paraf = hide_left_paraf
    global_hide_right_paraf = hide_right_paraf
    global_parafovea_width_pix = parafovea_pix
    global_det_max_slope = det_max_slope
    became_aligned = False
    if align_to_RPE != global_state_align2rpe:
        global_state_align2rpe = align_to_RPE
        do_update_annotations = True
        if align_to_RPE:
            became_aligned = True
            toggle_viz_unaligned_to_aligned(layers_unaligned, layers_rpe_align, user_points)
        else:
            toggle_viz_aligned_to_unaligned(layers_unaligned, layers_rpe_align, user_points)
        if user_points is not None:
            global_fovea_clicks = np.copy(user_points.data)
    elif mode == OCTAnnotationMode.Trace_Layers or (retina_layer is not None and retina_layer != last_selected_layer_label):
        if retina_layer in (RetinaLayers.RPE, RetinaLayers.ILM):
            oct_widget.align_to_RPE.value = False # force RPE/ILM annotations to be drawn in unaligned mode
            napari.current_viewer().layers.selection.select_only(user_rpe_annos)
        elif align_to_RPE:
            napari.current_viewer().layers.selection.select_only(layers_rpe_align.user_labels)
        else:
            napari.current_viewer().layers.selection.select_only(layers_unaligned.user_labels)
    elif mode is not None:
        napari.current_viewer().layers.selection.select_only(user_points)
    if retina_layer is not None:
        last_selected_layer_label = retina_layer
        if eraser:
            user_rpe_annos.mode = 'ERASE'
            layers_rpe_align.user_labels.mode = 'ERASE'
            layers_unaligned.user_labels.mode = 'ERASE'
            brushsize = 9
            user_rpe_annos.brush_size = brushsize
            layers_rpe_align.user_labels.brush_size = brushsize
            layers_unaligned.user_labels.brush_size = brushsize
        else:
            user_rpe_annos.mode = 'PAINT'
            layers_rpe_align.user_labels.mode = 'PAINT'
            layers_unaligned.user_labels.mode = 'PAINT'
            user_rpe_annos.selected_label = retina_layer
            layers_rpe_align.user_labels.selected_label = retina_layer
            layers_unaligned.user_labels.selected_label = retina_layer
            brushsize = 1
            user_rpe_annos.brush_size = brushsize
            layers_rpe_align.user_labels.brush_size = brushsize
            layers_unaligned.user_labels.brush_size = brushsize
    user_points.opacity = label_viz
    parafovea_indicator_lines.opacity = parafovea_indicator_opacity * label_viz
    if align_to_RPE:
        user_rpe_annos.opacity = 0
        layers_unaligned.image_layer.opacity = 0
        layers_unaligned.user_labels.opacity = 0
        layers_unaligned.auto_labels.opacity = 0
        layers_rpe_align.image_layer.opacity = 1
        layers_rpe_align.user_labels.opacity = label_viz * user_label_opacity
        layers_rpe_align.auto_labels.opacity = label_viz * autodets_viz * machine_labels_opacity
    else:
        user_rpe_annos.opacity = label_viz * user_label_opacity
        layers_unaligned.image_layer.opacity = 1
        layers_unaligned.user_labels.opacity = label_viz * user_label_opacity
        layers_unaligned.auto_labels.opacity = label_viz * autodets_viz * machine_labels_opacity
        layers_rpe_align.image_layer.opacity = 0
        layers_rpe_align.user_labels.opacity = 0
        layers_rpe_align.auto_labels.opacity = 0
    if became_aligned and retina_layer in (RetinaLayers.RPE, RetinaLayers.ILM):
        oct_widget.retina_layer.value = RetinaLayers.OPL
    if do_update_annotations:
        update_auto_detection_worker_send_user_annotations()
        if global_fovea_clicks is not None and len(global_fovea_clicks) == 3:
            update_parafovea_indicator_lines()


# The user shouldn't be able to edit the calculated metrics... they're read-only values
oct_widget.metrics.read_only = True

# Create a button to save a spreadsheet, and add it to the sidebar widget.
spreadsheet_save_button = widgets.PushButton(text="Save spreadsheet")
spreadsheet_save_button.changed.connect(build_spreadsheet_saver_function(oct_widget.metrics.data, metrics_table_rows))
oct_widget.append(spreadsheet_save_button)


# Send updated user annotations data to background computations worker thread
def update_auto_detection_worker_send_user_annotations():
    global user_rpe_annos, layers_unaligned, layers_rpe_align, auto_worker, worker_alive, global_fovea_clicks, global_parafovea_width_pix, global_hide_left_paraf, global_hide_right_paraf, global_det_max_slope
    if user_rpe_annos is None or layers_unaligned is None or layers_rpe_align is None or auto_worker is None or not worker_alive.is_set():
        return
    # copies of data should be made for multithreading safety
    if len(layers_unaligned.user_labels.data.shape) == 2:
        auto_worker.send((
            user_rpe_annos.data,
            layers_unaligned.user_labels.data,
            layers_rpe_align.user_labels.data,
            None,
            deepcopy(global_fovea_clicks),
            deepcopy(global_state_align2rpe),
            deepcopy(global_parafovea_width_pix),
            global_hide_left_paraf,
            global_hide_right_paraf,
            global_det_max_slope,
            ))
    else:
        slice3d = napari.current_viewer().dims.current_step[0]
        auto_worker.send((
            user_rpe_annos.data[slice3d],
            layers_unaligned.user_labels.data[slice3d],
            layers_rpe_align.user_labels.data[slice3d],
            deepcopy(slice3d),
            deepcopy(global_fovea_clicks),
            deepcopy(global_state_align2rpe),
            deepcopy(global_parafovea_width_pix),
            global_hide_left_paraf,
            global_hide_right_paraf,
            global_det_max_slope,
            ))


# relevant in 3D mode when viewing many images... function is called when bottom slider changes to a new image
def on_viewer_slice_step_change(event: Event):
    update_auto_detection_worker_send_user_annotations()


# Called whenever user scribbles on the image with the paintbrush. This callback is registered with the viewer in code below that looks like *.events*.connect(...)
def on_user_label_paint(event: Event):
    global user_rpe_annos, layers_unaligned, layers_rpe_align
    if not hasattr(event,'source'):
        return
    if event.source.name not in (user_rpe_annos.name, layers_unaligned.user_labels.name, layers_rpe_align.user_labels.name):
        return
    update_auto_detection_worker_send_user_annotations()

def update_parafovea_indicator_lines():
    global global_fovea_clicks, global_parafovea_width_pix, parafovea_indicator_opacity
    fovea_x = np.median(global_fovea_clicks[:,1])
    oldlines = np.float32(parafovea_indicator_lines.data)
    oldlines[0,:2,1] = fovea_x - global_parafovea_width_pix - radius1_cal
    oldlines[0,2:,1] = fovea_x - global_parafovea_width_pix + radius1_cal
    oldlines[1,:2,1] = fovea_x + global_parafovea_width_pix - radius1_cal
    oldlines[1,2:,1] = fovea_x + global_parafovea_width_pix + radius1_cal
    parafovea_indicator_lines.data = oldlines
    parafovea_indicator_opacity = 0.15
    parafovea_indicator_lines.opacity = oct_widget.label_viz.value

# Called when the user clicks a point.
def on_user_points_update(event: Event):
    global global_fovea_clicks, parafovea_indicator_lines, parafovea_indicator_opacity, global_parafovea_width_pix
    if not hasattr(event,'value') or not hasattr(event,'source'):
        return
    if len(event.source.data) > 3: # don't allow there to be a 4th clicked point... reset to no points
        assert len(event.source.data.shape) == 2 and int(event.source.data.shape[1]) == 2, str(event.source.data.shape)
        event.source.data = np.zeros((0,2), dtype=event.source.data.dtype)
    if len(event.source.data) == 3:
        global_fovea_clicks = np.copy(event.source.data)
        update_parafovea_indicator_lines()
        update_auto_detection_worker_send_user_annotations() # upon 3rd clicked point, update calculations
    else:
        parafovea_indicator_opacity = 0
        parafovea_indicator_lines.opacity = 0
        oct_widget.metrics.data = np.zeros((len(metrics_table_rows),1), np.int32)
        global_fovea_clicks = None


# Workaround for selecting newly created draw layer as soon as possible after on_layers_inserted
# "As soon as possible" = as soon as the user moves the mouse
def check_auto_select_draw_layer_after_image_open():
    global just_created_user_labels, user_rpe_annos, worker_alive
    if just_created_user_labels and worker_alive.is_set():
        just_created_user_labels = False
        napari.current_viewer().layers.selection.select_only(user_rpe_annos)
        update_auto_detection_worker_send_user_annotations()
def on_viewer_cursor_position(_):
    check_auto_select_draw_layer_after_image_open()
def on_viewer_mouse_over_canvas(_):
    check_auto_select_draw_layer_after_image_open()


# Called when a new layer is created... the first time this happens, it should be because a user opened an image.
def build_on_layers_inserted_for_thread_loop(thread_loop_func):
    def on_layers_inserted(event: Event):
        global user_rpe_annos, user_points, layers_unaligned, layers_rpe_align, parafovea_indicator_lines, auto_worker, just_created_user_labels, worker_alive
        if layers_unaligned is not None: # this would be called recursively as new layers are created... skip for any later layers after the first new one (which was the user-opened one)
            return
        if isinstance(event.source, LayerList) and len(event.source) == 1:
            layer, = event.source
        else:
            layer = event.source
        if not isinstance(layer, Image) or not hasattr(layer.data, 'shape'):
            return

        imshape = image_shape_to_2d_label_shape(layer.data.shape)
        if imshape is None:
            return

        viewer = napari.current_viewer()
        fovea_colors = {fvl.value: getattr(RetinaLayerColors, fvl.name).value for fvl in RetinaLayers}

        if len(imshape) == 2:
            globals.current_filename = re.sub(r'[^A-Za-z0-9_ \.]+', '', os.path.basename(str(layer.name))).strip()
            viewer.title = f"napari: {globals.current_filename}"
        else:
            globals.current_filename = "" # TODO: multiple filenames when multiple images are opened... need to modify napari reader to provide filenames in metadata

        print(f"creating labels of shape {imshape} for image of shape {layer.data.shape}")
        layers_unaligned = LayersGroup()
        layers_rpe_align = LayersGroup()

        # Creates a bunch of napari layers... this app hides napari's layer controls tab to de-clutter the interface (since so many new layers are created)
        layers_unaligned.image_layer = layer
        layers_rpe_align.image_layer = viewer.add_image(
            np.zeros(imshape[-2:], dtype=np.uint8),
            name='Aligned image', opacity=0)

        layers_unaligned.auto_labels = viewer.add_labels(
            np.zeros(imshape, dtype=np.int32),
            name='Machine detected labels', opacity=machine_labels_opacity,
            color=fovea_colors)
        layers_rpe_align.auto_labels = viewer.add_labels(
            np.zeros(imshape[-2:], dtype=np.int32),
            name='Aligned machine labels', opacity=0,
            color=fovea_colors)

        user_rpe_annos = viewer.add_labels(
            np.zeros(imshape, dtype=np.int32),
            name='User drawn RPE', opacity=user_label_opacity,
            color=fovea_colors)

        layers_unaligned.user_labels = viewer.add_labels(
            np.zeros(imshape, dtype=np.int32),
            name='User drawn labels', opacity=user_label_opacity,
            color=fovea_colors)
        layers_rpe_align.user_labels = viewer.add_labels(
            np.zeros(imshape, dtype=np.int32),
            name='Aligned user labels', opacity=0,
            color=fovea_colors)

        parafovea_indicator_lines = viewer.add_shapes(
            np.float32([[[0,0],[imshape[-2],0],[imshape[-2],0],[0,0]],[[0,0],[imshape[-2],0],[imshape[-2],0],[0,0]]]),
            shape_type="rectangle", edge_width=1, edge_color="yellow", face_color="yellow",
            name='Parafovea indicator lines', opacity=0)

        user_points = viewer.add_points(name='User clicked fovea points', opacity=1)

        auto_worker = thread_loop_func(np.array(layers_unaligned.image_layer.data).copy(), worker_alive)
        auto_worker.yielded.connect(
            build_on_yielded(auto_worker,
                layers_unaligned.auto_labels,
                layers_rpe_align.auto_labels,
                layers_rpe_align.image_layer,
                oct_widget.metrics)
            )
        auto_worker.start()

        user_points.mode = 'ADD'
        user_points.events.set_data.connect(on_user_points_update)
        user_points.events.data.connect(on_user_points_update)

        user_rpe_annos.mode = 'PAINT'
        user_rpe_annos.brush_size = 1
        user_rpe_annos.events.paint.connect(on_user_label_paint)

        for lgroup in [layers_unaligned, layers_rpe_align]:
            lgroup.user_labels.mode = 'PAINT'
            lgroup.user_labels.brush_size = 1
            lgroup.user_labels.events.paint.connect(on_user_label_paint)

        viewer.window.add_dock_widget(oct_widget)

        # It's not possible to select just-created labels layer from within this event callback
        # So, do it later, hooked to some other frequent event, like related to mouse movement
        just_created_user_labels = True

    return on_layers_inserted


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.qt_viewer.dockLayerList.setVisible(False) # de-clutter interface by hiding layer controls tab... makes interface cleaner since so many layers get created
    viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    viewer.layers.events.inserted.connect(build_on_layers_inserted_for_thread_loop(thread_work_loop_classic_detection))
    viewer.cursor.events.position.connect(on_viewer_cursor_position)
    viewer.events.mouse_over_canvas.connect(on_viewer_mouse_over_canvas)
    viewer.dims.events.current_step.connect(on_viewer_slice_step_change)
    napari.run()