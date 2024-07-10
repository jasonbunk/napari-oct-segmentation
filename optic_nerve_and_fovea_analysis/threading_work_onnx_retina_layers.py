# author: Jason Bunk
# This is the background thread for computations for the retina layers tracing application
import time
import os
thispath = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from napari.qt.threading import thread_worker
from onnx_detection import onnx_weights_path, preprocess_image_for_onnx_inference, smooth_onnx_detections, masked_sequential_detection_with_user_annos
from definitions_and_config import (
    RetinaLayers,
    retinalayers_metrics_table_rows,
    typical_image_width_pixels,
    typical_image_width_microns,
    typical_image_height_pixels,
    typical_image_height_microns,
    parafovea_radius_microns,
    radius_parafovea_calculations,
    radius_foveacenter_calculations,
)
from my_utils import make_3or4channel_image_grayscale
import globals # only access globals in the main thread, unless you are careful with a threading.Lock()
import onnxruntime

microns_per_pixel_horizontal = typical_image_width_microns / typical_image_width_pixels
microns_per_pixel_vertical = typical_image_height_microns / typical_image_height_pixels


# using napari qt threading because we want to connect the "yield" result to updating the interface ASAP
#     do that like this:
#         worker = thread_work_loop()
#         worker.yielded.connect(build_on_yielded(worker, ...))
#         worker.start()
# incoming tasks should not be queued (it would need to be a leaky queue with size 1)


# this function is run in the main thread of the main app
# create a callback function that is executed by the main thread whenever the background thread yields a value
def build_on_yielded(worker, auto_labels_raw, display_table):
    def on_yielded(value): # this is also run in the main thread of the main app
        if value is not None:
            whichslice, auto_labels_raw_slice, metrics_data = value
            auto_labels_raw_3d = np.copy(auto_labels_raw.data)
            auto_labels_raw_3d[whichslice] = auto_labels_raw_slice
            auto_labels_raw.data = auto_labels_raw_3d
            display_table.data = metrics_data
    return on_yielded


# This is run in a worker thread, separate from the main app.
# All communication with the main thread should be handled carefully in one of these ways:
#     * the initial argument to this function...
#         * passed as copy
#         * shared through threading data structure (Lock, Event, etc)
#     * passed back and forth via send/yield
#         * need to send/yield copies of data
@thread_worker
def thread_work_loop(image, worker_alive):
    yield None # need to do this once just before setting worker to ready state
    worker_alive.set() # now this worker is ready to receive annotations
    image, is2d = make_3or4channel_image_grayscale(image)

    session = onnxruntime.InferenceSession(onnx_weights_path, providers=['CPUExecutionProvider',])

    if is2d:
        assert len(image.shape) == 2, str(image.shape)
        medf = np.expand_dims(preprocess_image_for_onnx_inference(image), axis=0)
        (det_dnn,) = session.run(["output",], {"input":medf})
        det_dnn = (smooth_onnx_detections(det_dnn)*(-1.0)).squeeze(0).transpose(0,2,1)
    else:
        assert len(image.shape) == 3, str(image.shape)
        medf = np.stack([preprocess_image_for_onnx_inference(image[ii]) for ii in range(len(image))])

    results = None
    while worker_alive.is_set():
        received = yield results # primary communication between this worker and the main thread
        if received is None:
            time.sleep(0.001) # this is needed so that this worker thread doesn't hog the CPU and freeze the main app
            results = None
            continue

        unaligned_annos, slice_3d, fovea_click, det_max_slope = received
        unaligned_annos = np.array(unaligned_annos)

        if not is2d:
            (det_dnn,) = session.run(["output",], {"input":medf[slice_3d:slice_3d+1]})
            det_dnn = (smooth_onnx_detections(det_dnn)*(-1.0)).squeeze(0).transpose(0,2,1)

        det_user = masked_sequential_detection_with_user_annos(det_dnn, unaligned_annos,
                                                               ilm_nms_w=15, rpe_top_mask=21, ilm_top_mask=4, max_slope=max(1,det_max_slope))

        # not "anti-aliased", so if you zoom in, the line may look like disconnected dots
        computed_annos = np.zeros(image.shape[-2:], dtype=unaligned_annos.dtype)
        cols = np.arange(computed_annos.shape[1])
        name2idx = {}
        for fvl in RetinaLayers:
            name2idx[fvl.name] = fvl.value-1
            if fvl.name not in ("OPL", "CSJ"): # hide OPL because it's not needed for measurements below
                computed_annos[det_user[:,fvl.value-1], cols] = fvl.value

        metr = [[0,] for _ in range(len(retinalayers_metrics_table_rows))]
        mr = {vv:ii for (ii,vv) in enumerate(retinalayers_metrics_table_rows)} # mr["x"] is easier to type than metrics_table_rows.index("x")

        if fovea_click is not None:
            fovea_x = float(fovea_click[0,1])

            stat = lambda x_: np.mean(x_) # measurements will be mean thickness... can change this to median for robustness to layer trace errors
            iroc = det_user.astype(np.float32)

            for region in ("left", "center", "right"):
                if region == "left":
                    rleft = fovea_x - (parafovea_radius_microns + radius_parafovea_calculations) / microns_per_pixel_horizontal
                    rrigt = fovea_x - (parafovea_radius_microns - radius_parafovea_calculations) / microns_per_pixel_horizontal
                elif region == "right":
                    rleft = fovea_x + (parafovea_radius_microns - radius_parafovea_calculations) / microns_per_pixel_horizontal
                    rrigt = fovea_x + (parafovea_radius_microns + radius_parafovea_calculations) / microns_per_pixel_horizontal
                else:
                    rleft = fovea_x - radius_foveacenter_calculations / microns_per_pixel_horizontal
                    rrigt = fovea_x + radius_foveacenter_calculations / microns_per_pixel_horizontal

                rleft = max(0, int(round(rleft)))
                rrigt = min(iroc.shape[0], int(round(rrigt))+1)
                if rrigt > 0 and (rleft+1) < iroc.shape[0]:
                    region_thickn = lambda d: (stat(d[rleft:rrigt]) * microns_per_pixel_vertical)

                    metr[mr["RNFL_"+region]]   = region_thickn(iroc[:,name2idx["RNFLGCL"]] - iroc[:,name2idx["ILM"]])
                    metr[mr["GCL_"+region]]    = region_thickn(iroc[:,name2idx["GCLIPL"]]  - iroc[:,name2idx["RNFLGCL"]])
                    metr[mr["IPL_"+region]]    = region_thickn(iroc[:,name2idx["IPLINL"]]  - iroc[:,name2idx["GCLIPL"]])
                    metr[mr["RPE-ILM_"+region]]= region_thickn(iroc[:,name2idx["RPE"]]     - iroc[:,name2idx["ILM"]])
                else:
                    # region is entirely outside the image... can't calculate!
                    for rnm in ("RNFL_", "GCL_", "IPL_", "RPE-ILM_"):
                        metr[mr[rnm+region]] = -1

            # for napari spreadsheet, each metric needs to be a list of values; there's only one value
            metr = list([[vv,] for vv in metr])

        # results will be returned to the main thread via yield
        results = (slice_3d, computed_annos, metr)
