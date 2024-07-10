# Napari application for analyzing optic nerve disk
# author: Jason Bunk
import os
import re
from copy import deepcopy
import numpy as np
import napari
from magicgui import magicgui, widgets
from napari.layers import Image
from napari.components import LayerList
from napari.utils.events import Event
from my_utils import image_shape_to_2d_label_shape, make_3or4channel_image_grayscale, build_spreadsheet_saver_function, describe
from onnx_detection import onnx_weights_path, preprocess_image_for_onnx_inference, smooth_onnx_detections, masked_irx_detection_with_user_annos
from shapely import LineString, MultiLineString, Polygon
import onnxruntime
from definitions_and_config import (
    RetinaLayers,
    DiskRetinaLayers,
    RetinaLayerColors,
    OCTAnnotationMode,
    DiscClickMode,
    DiskClickColors,
    IndicatorLineTypes,
    IndicatorLineTypeColors,
    disc_metrics_table_rows,
    machine_labels_opacity,
    user_label_opacity,
    typical_image_width_pixels,
    typical_image_width_microns,
    typical_image_height_pixels,
    typical_image_height_microns,
    disc_paraside_microns,
    disc_paraside_radius1cal_microns,
)
import globals

just_created_user_labels = False
last_selected_layer_label = None
indicator_lines = None
user_points = None
user_drawn_layer_lines = None
machine_detected_layer_lines = None
onnx_detection_fields = None
image_width_pixels = None
image_height_pixels = None
rim_height_microns = None
last_rim_height = None
paraside_width_microns = None
last_paraside_width_microns = None
paraside_indicator_region = None
paraside_indicator_opacity = 0
do_physical_angles = True
last_do_physical_calc = True
microns_per_pixel_horizontal = typical_image_width_microns / typical_image_width_pixels
microns_per_pixel_vertical = typical_image_height_microns / typical_image_height_pixels
paraside_radius1cal_pixels = disc_paraside_radius1cal_microns / microns_per_pixel_horizontal

shown_retina_layers_idx_from_zero = tuple((int(fvl.value)-1 for fvl in DiskRetinaLayers))

print(f"microns_per_pixel_horizontal {microns_per_pixel_horizontal}")
print(f"microns_per_pixel_vertical {microns_per_pixel_vertical}")

# This configures the buttons and stuff on the right side of the app.
# It's a magicgui decorated function which is called whenever the user interacts with a button or slider.
# The decorator specifies the type of each widget, and initial values.
# The function decides what to do based on the widget state... for example, change the paintbrush size if needed.
@magicgui(auto_call=True,
    label_visibility={"widget_type":"FloatSlider", "max":1.0, "value":1.0},
    autodets_visibility={"widget_type":"FloatSlider", "max":1.0, "value":1.0},
    paraside_width={"widget_type":"FloatText", "value":disc_paraside_microns},
    rim_height={"widget_type":"FloatText", "value":150.0},
    mode={"widget_type":"RadioButtons", "value":OCTAnnotationMode.Trace_Layers},
    retina_layer={"widget_type":"RadioButtons", "value":DiskRetinaLayers.ILM},
    click_mode={"widget_type":"RadioButtons", "value":DiscClickMode.RPE_Endpoints},
    metrics={"widget_type":"Table", "value":{"data":[[0,],]*len(disc_metrics_table_rows), "index":disc_metrics_table_rows, "columns":["value",]}},
    )
def oct_widget(
    label_visibility:float,
    autodets_visibility:float,
    paraside_width:float,
    rim_height:float,
    mode: OCTAnnotationMode,
    retina_layer: DiskRetinaLayers,
    click_mode: DiscClickMode,
    eraser: bool,
    metrics: dict,
    physical_angles: bool=True,
    ):
    """This docstring to provides helpful popup text when the mouse hovers over something in the sidebar widget.

    Parameters
    ----------
    label_visibility : float
        Show or hide markup annotations; 0 = hide, 1 = show
    autodets_visibility : float
        Show or hide machine-detected annotations; 0 = hide, 1 = show
    paraside_width : float
        Distance in microns from disk center to the middle of the side regions where layer thickness measurements are made
    rim_height : float
        Disk/cup vertical distance in microns
    mode : Enum
        What should mouse click do? Either trace layer corrections; or click key points around the disk.
    retina_layer : Enum
        Which layer to trace/correct?
    click_mode : Enum
        If clicking mode, which type of point to click?
    eraser : bool
        If trace mode, use the mouse to erase line annotations.
    metrics : Table
        Table of measurements. Won't be valid without clicking disk points. Distance unit: microns.
    physical_angles : bool
        Should perpendicular angles be made with respect to the physical anatomical coordinate system? Or if not physical, the pixel coordinate system?
    """
    global user_points, user_drawn_layer_lines, machine_detected_layer_lines, last_selected_layer_label, \
        rim_height_microns, last_rim_height, paraside_indicator_region, paraside_indicator_opacity, \
        paraside_width_microns, last_paraside_width_microns, do_physical_angles
    if user_points is None or user_drawn_layer_lines is None:
        return
    rim_height_microns = rim_height
    paraside_width_microns = paraside_width
    want_computations = (rim_height_microns != last_rim_height or paraside_width_microns != last_paraside_width_microns or physical_angles != do_physical_angles)
    last_rim_height = deepcopy(rim_height_microns)
    last_paraside_width_microns = deepcopy(paraside_width_microns)
    do_physical_angles = deepcopy(physical_angles)
    if mode == OCTAnnotationMode.Trace_Layers:
        napari.current_viewer().layers.selection.select_only(user_drawn_layer_lines)
    else:
        napari.current_viewer().layers.selection.select_only(user_points[click_mode.value])

    if eraser:
        user_drawn_layer_lines.mode = 'ERASE'
        brushsize = 25
    else:
        user_drawn_layer_lines.selected_label = retina_layer
        user_drawn_layer_lines.mode = 'PAINT'
        brushsize = 1
    user_drawn_layer_lines.brush_size = brushsize

    for upl in user_points.values():
        upl.opacity = label_visibility
    indicator_lines.opacity = label_visibility * autodets_visibility
    paraside_indicator_region.opacity = label_visibility * autodets_visibility * paraside_indicator_opacity
    user_drawn_layer_lines.opacity = label_visibility * user_label_opacity
    machine_detected_layer_lines.opacity = label_visibility * autodets_visibility * machine_labels_opacity
    if want_computations:
        update_calculations_and_indicators_given_traced_lines(None)


# The user shouldn't be able to edit the calculated metrics... they're read-only values
oct_widget.metrics.read_only = True

# Create a button to save a spreadsheet, and add it to the sidebar widget.
spreadsheet_save_button = widgets.PushButton(text="Save spreadsheet")
spreadsheet_save_button.changed.connect(build_spreadsheet_saver_function(oct_widget.metrics.data, disc_metrics_table_rows))
oct_widget.append(spreadsheet_save_button)



def shapely_extended_line(pt1, pt2, add_length:float):
    assert pt1.size == 2 and pt2.size == 2, f"{pt1.shape}, {pt2.shape}"
    dir = (pt2 - pt1)
    dir *= add_length / max(1e-9, np.linalg.norm(dir))
    return LineString([(pt1 - dir).tolist(), (pt2 + dir).tolist()])
def shapely_line(pt1, pt2):
    assert pt1.size == 2 and pt2.size == 2, f"{pt1.shape}, {pt2.shape}"
    return LineString([pt1.tolist(), pt2.tolist()])
def sort_two_points_left_right(points_2x2):
    if points_2x2[0,1] <= points_2x2[1,1]:
        return points_2x2
    return np.stack((points_2x2[1,:], points_2x2[0,:]))
def line_len(a, b):
    assert len(a) == 2 and len(b) == 2, f"{len(a)} != 2 or {len(b)} != 2"
    return np.sqrt(np.square(float(a[0]) - float(b[0])) \
                 + np.square(float(a[1]) - float(b[1])))
def shapely_extract_intersection(isec):
    if isinstance(isec, MultiLineString):
        isec = list(isec.geoms)
        if len(isec) == 0:
            return np.empty((0,2), dtype=np.float64)
        if len(isec) == 1:
            isec = isec[0]
        else:
            return np.concatenate([shapely_extract_intersection(ls) for ls in isec])
    if isinstance(isec, LineString):
        isec = isec.xy
        if len(isec) > 0 and len(isec[0]) > 0:
            return np.stack(isec).transpose()
    return np.empty((0,2), dtype=np.float64)

def update_calculations_and_indicators_given_traced_lines(det_positions):
    global do_physical_angles, indicator_lines, user_points, image_width_pixels, image_height_pixels, rim_height_microns, paraside_indicator_region, paraside_indicator_opacity, paraside_width_microns
    assert microns_per_pixel_horizontal > 0 and microns_per_pixel_vertical > 0, f"h {microns_per_pixel_horizontal}, v {microns_per_pixel_vertical}"
    image_width_microns = image_width_pixels * microns_per_pixel_horizontal
    image_height_microns = image_height_pixels * microns_per_pixel_vertical
    indic = np.zeros_like(indicator_lines.data)
    paraside_center_x = None
    if len(user_points[DiscClickMode.RPE_Endpoints.value].data) == 2:
        image_diagonal_dist = np.sqrt(image_width_microns**2 + image_height_microns**2)
        line_extension_dist = abs(rim_height_microns) + image_diagonal_dist # long enough to go outside the image

        rpe_endpts = np.copy(user_points[DiscClickMode.RPE_Endpoints.value].data)
        if do_physical_angles:
            rpe_endpts[:,0] *= microns_per_pixel_vertical
            rpe_endpts[:,1] *= microns_per_pixel_horizontal
        else:
            rpe_endpts[:,0] /= microns_per_pixel_vertical
            rpe_endpts[:,1] /= microns_per_pixel_horizontal

        assert rpe_endpts[1,1] >= rpe_endpts[0,1], str(rpe_endpts) # rpe endpoints must have been sorted left-to-right
        rpe_dir = rpe_endpts[1] - rpe_endpts[0]
        rpe_dir /= np.linalg.norm(rpe_dir)
        rpe_normal = np.float64([-rpe_dir[1], rpe_dir[0]]) # because endpoints were sorted, this normal will point "up"
        assert rpe_normal[0] <= 0., f"{rpe_normal} ... {rpe_endpts}" # actually down, since y=0 is top of image, since it is row index

        if not do_physical_angles:
            rpe_endpts[:,0] *= microns_per_pixel_vertical**2
            rpe_endpts[:,1] *= microns_per_pixel_horizontal**2

        cup_line_inf = rpe_endpts + np.expand_dims(rpe_normal, axis=0) * rim_height_microns
        rpe_endp_cuplevel = np.copy(cup_line_inf)
        cup_line_inf = shapely_extended_line(cup_line_inf[0], cup_line_inf[1], line_extension_dist)

        if det_positions is None:
            det_positions = np.copy(machine_detected_layer_lines.data)
        lastcol = int(det_positions.shape[1])-1
        ilm_idx = shown_retina_layers_idx_from_zero.index(DiskRetinaLayers.ILM.value - 1)
        rnflgcl_idx = shown_retina_layers_idx_from_zero.index(DiskRetinaLayers.RNFLGCL.value - 1)

        empty_volume_above = np.float64([[-(rim_height_microns/microns_per_pixel_vertical+1), 0],] + det_positions[ilm_idx,:,:].tolist() + [[-(rim_height_microns/microns_per_pixel_vertical+1), lastcol],])
        empty_volume_above[:,0] *= microns_per_pixel_vertical
        empty_volume_above[:,1] *= microns_per_pixel_horizontal
        empty_volume_above = Polygon(empty_volume_above.tolist())
        cup_coll = shapely_extract_intersection(cup_line_inf.intersection(empty_volume_above))
        rpe_left_coll = shapely_extract_intersection(shapely_line(rpe_endpts[0], rpe_endpts[0] + rpe_normal * line_extension_dist).intersection(empty_volume_above))
        rpe_rigt_coll = shapely_extract_intersection(shapely_line(rpe_endpts[1], rpe_endpts[1] + rpe_normal * line_extension_dist).intersection(empty_volume_above))

        #if len(cup_coll) > 0 and len(cup_coll) % 2 == 0 and len(rpe_left_coll) > 0 and len(rpe_rigt_coll) > 0:
        if len(cup_coll) == 0 or len(cup_coll) % 2 != 0 or len(rpe_left_coll) == 0 or len(rpe_rigt_coll) == 0:
            if len(cup_coll) == 0:
                print("warning: no cup collision")
            if len(cup_coll) % 2 != 0:
                print(f"warning: odd number of cup collisions: {len(cup_coll)}")
            if len(rpe_left_coll) == 0:
                print("warning: no collision above left rpe endpoint")
            if len(rpe_rigt_coll) == 0:
                print("warning: no collision above right rpe endpoint")
        else:
            # cup is the largest/widest uninterrupted span
            spans = (np.square(cup_coll[1::2,0]-cup_coll[::2,0]) + np.square(cup_coll[1::2,1]-cup_coll[::2,1]))
            assert len(spans.shape) == 1, str(spans.shape)
            # it must appear between the RPE endpoints
            numvalid = len(spans)
            for ii in range(len(spans)):
                if cup_coll[ii*2:(ii+1)*2,1].max() < rpe_endpts[:,1].min() or cup_coll[ii*2:(ii+1)*2,1].min() > rpe_endpts[:,1].max():
                    spans[ii] /= line_extension_dist
                    numvalid -= 1
            if numvalid <= 0:
                print("warning (2): no cup collision")
            else:
                if len(cup_coll) > 2:
                    spans = np.argsort(spans)
                    cup_coll = cup_coll[spans[-1]*2:spans[-1]*2+2]
                cup_coll = sort_two_points_left_right(cup_coll)

                indic[IndicatorLineTypes.RPE_Endpoints.value] = rpe_endpts

                # extract first (lowest) collision
                if len(rpe_left_coll) == 1:
                    rpe_left_coll = rpe_left_coll[0]
                else:
                    rpe_left_coll = rpe_left_coll[np.argsort(rpe_left_coll[:,0])[-1]]
                if len(rpe_rigt_coll) == 1:
                    rpe_rigt_coll = rpe_rigt_coll[0]
                else:
                    rpe_rigt_coll = rpe_rigt_coll[np.argsort(rpe_rigt_coll[:,0])[-1]]

                # draw blue lines going upward from rpe endpoints to the ILM... if they collided
                if rpe_endp_cuplevel[0,0] >= rpe_left_coll[0]:
                    indic[IndicatorLineTypes.Rim_Left.value ] = np.stack((rpe_endp_cuplevel[0], rpe_left_coll))
                    rpe_cupcoll_left_valid = True
                else:
                    rpe_cupcoll_left_valid = False
                if rpe_endp_cuplevel[1,0] >= rpe_rigt_coll[0]:
                    indic[IndicatorLineTypes.Rim_Right.value] = np.stack((rpe_endp_cuplevel[1], rpe_rigt_coll))
                    rpe_cupcoll_rigt_valid = True
                else:
                    rpe_cupcoll_rigt_valid = False

                # Set indicator line to extrapolated intersection points
                indic[IndicatorLineTypes.Cup_Rim.value] = cup_coll

                cupbot = np.copy(user_points[DiscClickMode.Cup_Bottom.value].data)
                if cupbot.size > 0 and cupbot.max() > 1e-9:
                    cupbot = cupbot.flatten()
                    cupbot[0] *= microns_per_pixel_vertical
                    cupbot[1] *= microns_per_pixel_horizontal

                    cupcentroid = cup_coll.mean(axis=0) # middle of red line

                    buildthru = lambda pt: np.stack((pt - rpe_normal * image_diagonal_dist, pt + rpe_normal * image_diagonal_dist), axis=0)
                    parathru_ll = buildthru(cupcentroid - rpe_dir * (paraside_width_microns + disc_paraside_radius1cal_microns))
                    parathru_lr = buildthru(cupcentroid - rpe_dir * (paraside_width_microns - disc_paraside_radius1cal_microns))[::-1]
                    parathru_rl = buildthru(cupcentroid + rpe_dir * (paraside_width_microns - disc_paraside_radius1cal_microns))
                    parathru_rr = buildthru(cupcentroid + rpe_dir * (paraside_width_microns + disc_paraside_radius1cal_microns))[::-1]

                    rnfl_volume = np.concatenate((det_positions[ilm_idx,:,:], det_positions[rnflgcl_idx,:,:][::-1,:]), axis=0)
                    rnfl_volume[:,0] *= microns_per_pixel_vertical
                    rnfl_volume[:,1] *= microns_per_pixel_horizontal
                    rnfl_volume = Polygon(rnfl_volume.tolist())

                    parathru_leftvol = Polygon(parathru_ll.tolist() + parathru_lr.tolist()).intersection(rnfl_volume)
                    parathru_rigtvol = Polygon(parathru_rl.tolist() + parathru_rr.tolist()).intersection(rnfl_volume)

                    cup_dir = shapely_extended_line(cupbot, cupbot + rpe_normal, line_extension_dist)
                    cup_intersection = cup_dir.intersection(cup_line_inf)
                    if cup_intersection is None:
                        print(f"cup bottom upward does not intersect?")
                    else:
                        cup_intersection = cup_intersection.xy
                        assert len(cup_intersection) == 2, str(cup_intersection)
                        cup_intersection = np.float64(cup_intersection).flatten()
                        assert cup_intersection.size == 2, str(cup_intersection)

                        # Set indicator line end point to intersection
                        indic[IndicatorLineTypes.Cup_Depth.value] = np.stack((cupbot, cup_intersection))

                        # paraside visualization
                        paraside_center_x = np.concatenate((rpe_endpts[:,1], cupbot[1:2])).mean()
                        oldlines = np.float32(paraside_indicator_region.data)
                        oldlines[0,:2,:] = parathru_ll
                        oldlines[0,2:,:] = parathru_lr
                        oldlines[1,:2,:] = parathru_rl
                        oldlines[1,2:,:] = parathru_rr

                        # convert back from microns to pixels
                        oldlines[:,:,0] /= microns_per_pixel_vertical
                        oldlines[:,:,1] /= microns_per_pixel_horizontal

                        paraside_indicator_region.data = oldlines
                        paraside_indicator_opacity = 0.15
                        paraside_indicator_region.opacity = oct_widget.label_visibility.value * paraside_indicator_opacity

                        # finally, update metrics display
                        metr = np.zeros((len(disc_metrics_table_rows),), np.float32)
                        mr = {vv:ii for (ii,vv) in enumerate(disc_metrics_table_rows)} # mr["x"] is easier to type than metrics_table_rows.index("x")
                        metr[mr["disc_diameter"]] = line_len(rpe_endpts[0], rpe_endpts[1])
                        metr[mr["cup_diameter"]] = line_len(cup_coll[0], cup_coll[1])
                        metr[mr["max_cup_depth"]] = line_len(cupbot, cup_intersection)
                        metr[mr["neural_rim_left"]] = line_len(rpe_endp_cuplevel[0], cup_coll[0])
                        metr[mr["neural_rim_right"]] = line_len(rpe_endp_cuplevel[1], cup_coll[1])
                        metr[mr["neural_height_left"]] = line_len(rpe_endp_cuplevel[0], rpe_left_coll) if rpe_cupcoll_left_valid else 0.
                        metr[mr["neural_height_right"]] = line_len(rpe_endp_cuplevel[1], rpe_rigt_coll) if rpe_cupcoll_rigt_valid else 0.
                        metr[mr["RNFL_avgthk_peri_left"]] = parathru_leftvol.area / (disc_paraside_radius1cal_microns * 2)
                        metr[mr["RNFL_avgthk_peri_right"]] = parathru_rigtvol.area / (disc_paraside_radius1cal_microns * 2)
                        oct_widget.metrics.data = np.expand_dims(metr,axis=1)

    # convert back from microns to pixels
    indic[:,:,0] /= microns_per_pixel_vertical
    indic[:,:,1] /= microns_per_pixel_horizontal

    indicator_lines.data = indic
    if paraside_center_x is None:
        paraside_indicator_opacity = 0
        paraside_indicator_region.opacity = 0
        oct_widget.metrics.data = np.zeros((len(disc_metrics_table_rows),1), np.int32)


def update_line_traces_then_update_calculations_and_indicators():
    global indicator_lines, user_points, user_drawn_layer_lines, machine_detected_layer_lines, onnx_detection_fields
    if user_points is None or user_drawn_layer_lines is None:
        return
    if len(user_drawn_layer_lines.data.shape) == 2:
        detslice = onnx_detection_fields.copy().squeeze(0)
        user_annos_slice = user_drawn_layer_lines.data.copy()
    else:
        slice3d = napari.current_viewer().dims.current_step[0]
        detslice = onnx_detection_fields[slice3d].copy()
        user_annos_slice = user_drawn_layer_lines.data[slice3d].copy()

    iroc_raw = masked_irx_detection_with_user_annos(detslice, user_annos_slice, shown_retina_layers_idx_from_zero,
                                                    ilm_nms_w=15, rpe_top_mask=21, ilm_top_mask=4, max_slope=20)
    det_positions = np.copy(machine_detected_layer_lines.data)
    for ii in range(det_positions.shape[0]):
        det_positions[ii,:,0] = iroc_raw[:,shown_retina_layers_idx_from_zero[ii]]
    machine_detected_layer_lines.data = det_positions
    update_calculations_and_indicators_given_traced_lines(det_positions)


# Called when the user clicks a point.
def on_user_points_update(event: Event):
    global indicator_lines, user_points, user_drawn_layer_lines
    if not hasattr(event,'value') or not hasattr(event,'source') or user_points is None:
        return
    if event.source.name == DiscClickMode.Cup_Bottom.name:
        event.source.data = np.copy(event.source.data[-1,:])
    else:
        assert event.source.name == DiscClickMode.RPE_Endpoints.name, str(event.source)
        if len(event.source.data) == 2:
            event.source.data = sort_two_points_left_right(np.copy(event.source.data))
        elif len(event.source.data) > 2:
            event.source.data = np.zeros((0,2), dtype=event.source.data.dtype)
    update_calculations_and_indicators_given_traced_lines(None)


# Called whenever user scribbles on the image with the paintbrush. This callback is registered with the viewer in code below that looks like *.events*.connect(...)
def on_user_label_paint(event: Event):
    global user_drawn_layer_lines
    if not hasattr(event,'source'):
        return
    if event.source.name != user_drawn_layer_lines.name:
        return
    update_line_traces_then_update_calculations_and_indicators()


# relevant in 3D mode when viewing many images... function is called when bottom slider changes to a new image
def on_viewer_slice_step_change(event: Event):
    if globals.opened_filenames is not None:
        viewer = napari.current_viewer()
        slice3d = viewer.dims.current_step[0]
        globals.current_filename = globals.opened_filenames[slice3d]
        viewer.title = f"napari: {globals.current_filename}"
    update_line_traces_then_update_calculations_and_indicators()


# Workaround for selecting newly created draw layer as soon as possible after on_layers_inserted
# "As soon as possible" = as soon as the user moves the mouse
def check_auto_select_draw_layer_after_image_open():
    global just_created_user_labels, user_drawn_layer_lines
    if just_created_user_labels and user_drawn_layer_lines is not None:
        just_created_user_labels = False
        napari.current_viewer().layers.selection.select_only(user_drawn_layer_lines)
        update_line_traces_then_update_calculations_and_indicators()
def on_viewer_cursor_position(_):
    check_auto_select_draw_layer_after_image_open()
def on_viewer_mouse_over_canvas(_):
    check_auto_select_draw_layer_after_image_open()


# Called when a new layer is created... the first time this happens, it should be because a user opened an image.
def on_layers_inserted(event: Event):
    global indicator_lines, user_points, user_drawn_layer_lines, machine_detected_layer_lines, image_width_pixels, image_height_pixels, onnx_detection_fields, just_created_user_labels, paraside_indicator_region
    if user_points is not None: # this would be called recursively as new layers are created... skip for any later layers after the first new one (which was the user-opened one)
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
    #else:
        #globals.opened_filenames = list([re.sub(r'[^A-Za-z0-9_ \.]+', '', os.path.basename(ll)).strip() for ll in layer.metadata["filenames"]]) # regex to sanitize for saving into .csv spreadsheet
        #globals.current_filename = globals.opened_filenames[0]
        #print(f"opened {len(globals.opened_filenames)} images")

    image_width_pixels = imshape[-1]
    image_height_pixels = imshape[-2]

    print(f"creating labels of shape {imshape} for image of shape {layer.data.shape} where image are shape {image_width_pixels} x {image_height_pixels}")
    indic_colors = list([cc.value for cc in IndicatorLineTypeColors])


    paraside_indicator_region = viewer.add_shapes(
        np.float32([[[0,0],[imshape[-2],0],[imshape[-2],0],[0,0]],[[0,0],[imshape[-2],0],[imshape[-2],0],[0,0]]]),
        shape_type="polygon", edge_width=1, edge_color="yellow", face_color="yellow",
        name='paraside indicator lines', opacity=0)

    user_drawn_layer_lines = viewer.add_labels(
        np.zeros(imshape, dtype=np.int32),
        name='User drawn layers', opacity=user_label_opacity,
        color=fovea_colors)

    det_positions = np.float32([[[0,0] for _ in range(image_width_pixels)] for _ in range(len(shown_retina_layers_idx_from_zero))])
    for ii in range(det_positions.shape[0]):
        det_positions[ii,:,1] = np.arange(det_positions.shape[1])
    path_colors = list([getattr(RetinaLayerColors, fvl.name).value for fvl in DiskRetinaLayers])
    machine_detected_layer_lines = viewer.add_shapes(
        det_positions,
        shape_type="path",
        edge_width=1, edge_color=path_colors,
        face_color=path_colors,
        name='Machine detected layers', opacity=machine_labels_opacity)

    indicator_lines = viewer.add_shapes(
        np.float32([[[0,0],[0,0]] for _ in range(len(IndicatorLineTypes))]),
        shape_type="line", edge_width=1,
        edge_color=indic_colors,
        face_color=indic_colors,
        name='Indicator lines', opacity=0)

    #viewer.add_labels(
    #    np.zeros(imshape, dtype=np.int32),
    #    name='Machine detected layers', opacity=machine_labels_opacity,
    #    color=fovea_colors)

    user_drawn_layer_lines.mode = 'PAINT'
    user_drawn_layer_lines.brush_size = 1
    user_drawn_layer_lines.events.paint.connect(on_user_label_paint)

    user_points = {}
    for clkm in DiscClickMode:
        user_points[clkm.value] = viewer.add_points(name=clkm.name, opacity=1, face_color=[getattr(DiskClickColors, clkm.name).value,])
        user_points[clkm.value].mode = 'ADD'
        user_points[clkm.value].events.set_data.connect(on_user_points_update)
        user_points[clkm.value].events.data.connect(on_user_points_update)

    viewer.window.add_dock_widget(oct_widget)

    # run ONNX neural network calculations once
    image, is2d = make_3or4channel_image_grayscale(layer.data)
    image = np.ascontiguousarray(image)
    session = onnxruntime.InferenceSession(onnx_weights_path, providers=['CPUExecutionProvider',])
    if is2d:
        assert len(image.shape) == 2, str(image.shape)
        (onnx_detection_fields,) = session.run(["output",], {"input":np.expand_dims(preprocess_image_for_onnx_inference(image),axis=0)})
    else:
        assert len(image.shape) == 3, str(image.shape)
        onnx_detection_fields = []
        for ii in range(len(image)):
            (result,) = session.run(["output",], {"input":np.expand_dims(preprocess_image_for_onnx_inference(image[ii]),axis=0)})
            onnx_detection_fields.append(result)
        onnx_detection_fields = np.concatenate(onnx_detection_fields)
    onnx_detection_fields = smooth_onnx_detections(onnx_detection_fields)
    onnx_detection_fields = np.moveaxis(-1.0*onnx_detection_fields, -2, -1)

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
