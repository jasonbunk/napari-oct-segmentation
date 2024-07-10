# Napari application for tracing retina layers and measuring thicknesses around the fovea center
# author: Jason Bunk
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
from threading_work_onnx_retina_layers import thread_work_loop, build_on_yielded
from my_utils import image_shape_to_2d_label_shape, build_spreadsheet_saver_function
from layer_group_alignment import LayersGroup
from definitions_and_config import (
    TraceableRetinaLayers,
    RetinaLayerColors,
    OCTAnnotationMode,
    machine_labels_opacity,
    user_label_opacity,
    retinalayers_metrics_table_rows,
    typical_image_width_pixels,
    typical_image_width_microns,
    parafovea_radius_microns,
    radius_parafovea_calculations,
    radius_foveacenter_calculations,
)
import globals

user_points = None
layers_unaligned = None
parafovea_indicator_lines = None
parafovea_indicator_opacity = 0

auto_worker = None
worker_alive = threading.Event() # used for multithreading safety. By default, state is False. The launched computations thread will set to true when it's ready.
just_created_user_labels = False
last_selected_layer_label = None
global_fovea_clicks = None # either contains clicked point, or is None
global_detection_max_slope = 3

microns_per_pixel_horizontal = typical_image_width_microns / typical_image_width_pixels
print(f"typical_image_width_microns {typical_image_width_microns}, typical_image_width_pixels {typical_image_width_pixels}, microns_per_pixel_horizontal {microns_per_pixel_horizontal}")


# This configures the buttons and stuff on the right side of the app.
# It's a magicgui decorated function which is called whenever the user interacts with a button or slider.
# The decorator specifies the type of each widget, and initial values.
# The function decides what to do based on the widget state... for example, change the paintbrush size if needed.
@magicgui(auto_call=True,
    label_visibility={"widget_type":"FloatSlider", "max":1.0, "value":1.0},
    autodets_visibility={"widget_type":"FloatSlider", "max":1.0, "value":1.0},
    detection_max_slope={"widget_type":"IntSlider", "max":25, "min":1, "value":4},
    retina_layer={"widget_type":"RadioButtons", "value":TraceableRetinaLayers.RPE},
    mode={"widget_type":"RadioButtons", "value":OCTAnnotationMode.Trace_Layers},
    metrics={"widget_type":"Table", "value":{"data":[[0,],]*len(retinalayers_metrics_table_rows), "index":retinalayers_metrics_table_rows, "columns":["value",]}})
def oct_widget(
    label_visibility:float,
    autodets_visibility:float,
    detection_max_slope,
    retina_layer: TraceableRetinaLayers,
    mode: OCTAnnotationMode,
    eraser: bool,
    metrics: dict
    ):
    """This docstring to provides helpful popup text when the mouse hovers over something in the sidebar widget.

    Parameters
    ----------
    label_visibility : float
        Show or hide markup annotations; 0 = hide, 1 = show
    autodets_visibility : float
        Show or hide machine-detected annotations; 0 = hide, 1 = show
    retina_layer : Enum
        Which layer to trace/correct?
    mode : Enum
        What should mouse click do? Either trace layer corrections; or click fovea points for fovea angle measurement.
    eraser : bool
        Click to use the mouse to erase your annotations.
    metrics : Table
        Table of measurements. Won't be valid without clicking fovea center. Distance unit: microns.
    """
    global user_points, layers_unaligned, last_selected_layer_label, parafovea_indicator_lines, parafovea_indicator_opacity, global_detection_max_slope
    if user_points is None or layers_unaligned is None:
        return
    if mode == OCTAnnotationMode.Trace_Layers or (retina_layer is not None and retina_layer != last_selected_layer_label):
        napari.current_viewer().layers.selection.select_only(layers_unaligned.user_labels)
    elif mode is not None:
        napari.current_viewer().layers.selection.select_only(user_points)
    if retina_layer is not None:
        last_selected_layer_label = retina_layer
        if eraser:
            layers_unaligned.user_labels.mode = 'ERASE'
            brushsize = 25
        else:
            layers_unaligned.user_labels.mode = 'PAINT'
            layers_unaligned.user_labels.selected_label = retina_layer
            brushsize = 1
        layers_unaligned.user_labels.brush_size = brushsize
    user_points.opacity = label_visibility
    parafovea_indicator_lines.opacity = parafovea_indicator_opacity * autodets_visibility * label_visibility
    layers_unaligned.image_layer.opacity = 1
    layers_unaligned.user_labels.opacity = label_visibility * user_label_opacity
    layers_unaligned.auto_labels.opacity = label_visibility * autodets_visibility * machine_labels_opacity
    if global_detection_max_slope != detection_max_slope:
        global_detection_max_slope = deepcopy(detection_max_slope)
        update_auto_detection_worker_send_user_annotations()


# The user shouldn't be able to edit the calculated metrics... they're read-only values
oct_widget.metrics.read_only = True

# Create a button to save a spreadsheet, and add it to the sidebar widget.
spreadsheet_save_button = widgets.PushButton(text="Save spreadsheet")
spreadsheet_save_button.changed.connect(build_spreadsheet_saver_function(oct_widget.metrics.data, retinalayers_metrics_table_rows))
oct_widget.append(spreadsheet_save_button)


# Send updated user annotations data to background computations worker thread
def update_auto_detection_worker_send_user_annotations():
    global layers_unaligned, auto_worker, worker_alive, global_fovea_clicks
    if layers_unaligned is None or auto_worker is None or not worker_alive.is_set():
        return
    # copies of data should be made for multithreading safety
    if len(layers_unaligned.user_labels.data.shape) == 2:
        auto_worker.send((
            layers_unaligned.user_labels.data,
            None,
            deepcopy(global_fovea_clicks),
            global_detection_max_slope,
            ))
    else:
        viewer = napari.current_viewer()
        slice3d = viewer.dims.current_step[0]
        auto_worker.send((
            layers_unaligned.user_labels.data[slice3d],
            deepcopy(slice3d),
            deepcopy(global_fovea_clicks),
            global_detection_max_slope,
            ))


# relevant in 3D mode when viewing many images... function is called when bottom slider changes to a new image
def on_viewer_slice_step_change(event: Event):
    if globals.opened_filenames is not None:
        viewer = napari.current_viewer()
        slice3d = viewer.dims.current_step[0]
        globals.current_filename = globals.opened_filenames[slice3d]
        viewer.title = f"napari: {globals.current_filename}"
    update_auto_detection_worker_send_user_annotations()


# Called whenever user scribbles on the image with the paintbrush. This callback is registered with the viewer in code below that looks like *.events*.connect(...)
def on_user_label_paint(event: Event):
    global layers_unaligned
    if not hasattr(event,'source'):
        return
    if event.source.name != layers_unaligned.user_labels.name:
        return
    update_auto_detection_worker_send_user_annotations()


# Called when the user clicks a point.
def on_user_points_update(event: Event):
    global global_fovea_clicks, parafovea_indicator_lines, parafovea_indicator_opacity
    if not hasattr(event,'value') or not hasattr(event,'source'):
        return
    if len(event.source.data) > 1: # don't allow there to be a second clicked point... reset to no points
        assert len(event.source.data.shape) == 2 and int(event.source.data.shape[1]) == 2, str(event.source.data.shape)
        event.source.data = np.zeros((0,2), dtype=event.source.data.dtype)
    if len(event.source.data) == 1:
        global_fovea_clicks = np.copy(event.source.data)
        fovea_x = float(global_fovea_clicks[0,1])
        oldlines = np.float32(parafovea_indicator_lines.data)
        oldlines[0,:2,1] = fovea_x - (parafovea_radius_microns - radius_parafovea_calculations) / microns_per_pixel_horizontal
        oldlines[0,2:,1] = fovea_x - (parafovea_radius_microns + radius_parafovea_calculations) / microns_per_pixel_horizontal
        oldlines[1,:2,1] = fovea_x + (parafovea_radius_microns - radius_parafovea_calculations) / microns_per_pixel_horizontal
        oldlines[1,2:,1] = fovea_x + (parafovea_radius_microns + radius_parafovea_calculations) / microns_per_pixel_horizontal
        oldlines[2,:2,1] = fovea_x - radius_foveacenter_calculations / microns_per_pixel_horizontal
        oldlines[2,2:,1] = fovea_x + radius_foveacenter_calculations / microns_per_pixel_horizontal
        parafovea_indicator_lines.data = oldlines
        parafovea_indicator_opacity = 0.15
        parafovea_indicator_lines.opacity = oct_widget.label_visibility.value
        update_auto_detection_worker_send_user_annotations() # upon 3rd clicked point, update calculations
    else:
        parafovea_indicator_opacity = 0
        parafovea_indicator_lines.opacity = 0
        oct_widget.metrics.data = np.zeros((len(retinalayers_metrics_table_rows),1), np.int32)
        global_fovea_clicks = None


# Workaround for selecting newly created draw layer as soon as possible after on_layers_inserted
# "As soon as possible" = as soon as the user moves the mouse
def check_auto_select_draw_layer_after_image_open():
    global just_created_user_labels, worker_alive, layers_unaligned
    if just_created_user_labels and worker_alive.is_set() and layers_unaligned is not None:
        just_created_user_labels = False
        napari.current_viewer().layers.selection.select_only(layers_unaligned.user_labels)
        update_auto_detection_worker_send_user_annotations()
def on_viewer_cursor_position(_):
    check_auto_select_draw_layer_after_image_open()
def on_viewer_mouse_over_canvas(_):
    check_auto_select_draw_layer_after_image_open()


# Called when a new layer is created... the first time this happens, it should be because a user opened an image.
def on_layers_inserted(event: Event):
    global user_points, layers_unaligned, parafovea_indicator_lines, auto_worker, just_created_user_labels, worker_alive
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
    fovea_colors = {fvl.value: getattr(RetinaLayerColors, fvl.name).value for fvl in TraceableRetinaLayers}

    if len(imshape) == 2:
        globals.current_filename = re.sub(r'[^A-Za-z0-9_ \.]+', '', os.path.basename(str(layer.name))).strip()
        viewer.title = f"napari: {globals.current_filename}"
    #else:
        #globals.opened_filenames = list([re.sub(r'[^A-Za-z0-9_ \.]+', '', os.path.basename(ll)).strip() for ll in layer.metadata["filenames"]]) # regex to sanitize for saving into .csv spreadsheet
        #globals.current_filename = globals.opened_filenames[0]
        #print(f"opened {len(globals.opened_filenames)} images")

    print(f"creating labels of shape {imshape} for image of shape {layer.data.shape}")
    layers_unaligned = LayersGroup()

    # Creates a bunch of napari layers... this app hides napari's layer controls tab to de-clutter the interface (since so many new layers are created)
    layers_unaligned.image_layer = layer

    layers_unaligned.auto_labels = viewer.add_labels(
        np.zeros(imshape, dtype=np.int32),
        name='Machine detected labels', opacity=machine_labels_opacity,
        color=fovea_colors)

    layers_unaligned.user_labels = viewer.add_labels(
        np.zeros(imshape, dtype=np.int32),
        name='User drawn labels', opacity=user_label_opacity,
        color=fovea_colors)

    parafovea_indicator_lines = viewer.add_shapes(
        np.float32([[[0,0],[imshape[-2],0],[imshape[-2],0],[0,0]] for _ in range(3)]),
        shape_type="rectangle", edge_width=1, edge_color="yellow", face_color="yellow",
        name='Parafovea indicator lines', opacity=0)

    user_points = viewer.add_points(name='User clicked fovea points', opacity=1)

    auto_worker = thread_work_loop(np.array(layers_unaligned.image_layer.data).copy(), worker_alive)
    auto_worker.yielded.connect(
        build_on_yielded(auto_worker,
            layers_unaligned.auto_labels,
            oct_widget.metrics)
        )
    auto_worker.start()

    user_points.mode = 'ADD'
    user_points.events.set_data.connect(on_user_points_update)
    user_points.events.data.connect(on_user_points_update)

    layers_unaligned.user_labels.mode = 'PAINT'
    layers_unaligned.user_labels.brush_size = 1
    layers_unaligned.user_labels.events.paint.connect(on_user_label_paint)

    viewer.window.add_dock_widget(oct_widget)

    # It's not possible to select just-created labels layer from within this event callback
    # So, do it later, hooked to some other frequent event, like related to mouse movement
    just_created_user_labels = True


viewer = napari.Viewer()
viewer.window.qt_viewer.dockLayerList.setVisible(False) # de-clutter interface by hiding layer controls tab... makes interface cleaner since so many layers get created
viewer.window.qt_viewer.dockLayerControls.setVisible(False)
viewer.layers.events.inserted.connect(on_layers_inserted)
viewer.cursor.events.position.connect(on_viewer_cursor_position)
viewer.events.mouse_over_canvas.connect(on_viewer_mouse_over_canvas)
viewer.dims.events.current_step.connect(on_viewer_slice_step_change)
napari.run()
