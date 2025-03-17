# Author: Jason Bunk
import time
import numpy as np
from napari.qt.threading import thread_worker
from oct_detection import (
    ridge_detect_opl_csj_given_user_annos,
    new_ilmrpe_detect_rpe_only_first,
    precompute_ilm_and_rpe_det_energy_maps,
    precompute_opl_csj_det_energy_maps,
    precompute_csj_energy_map,
    straighten_rpe,
)
from my_utils import make_3or4channel_image_grayscale
from layer_group_alignment import aligned_to_unaligned_2d
from onnx_detection import onnx_weights_path, irgior_to_iroc, preprocess_image_for_onnx_inference, smooth_onnx_detections, masked_sequential_iroc_with_rioc_user_annos
from calculate_paper_metrics import calculate_metrics
import globals # only access globals in the main thread, unless you are careful with a threading.Lock()


# using napari qt threading because we want to connect the "yield" result to updating the interface ASAP
#     do that like this:
#         worker = thread_work_loop()
#         worker.yielded.connect(build_on_yielded(worker, ...))
#         worker.start()
# incoming tasks should not be queued (it would need to be a leaky queue with size 1)


# this function is run in the main thread of the main app
# create a callback function that is executed by the main thread whenever the background thread yields a value
def build_on_yielded(worker, auto_labels_raw, auto_labels_reg, rpe_aligned_image, display_table):
    def on_yielded(value): # this is also run in the main thread of the main app
        if value is not None:
            whichslice, rpe_aligned_image.data, auto_labels_raw_slice, auto_labels_reg.data, globals.latest_detected_layers, metrics_data = value
            auto_labels_raw_3d = np.copy(auto_labels_raw.data)
            auto_labels_raw_3d[whichslice] = auto_labels_raw_slice
            auto_labels_raw.data = auto_labels_raw_3d
            display_table.data = metrics_data
    return on_yielded


kwidth_csj = 15
rpe_thickness = 56

# This is run in a worker thread, separate from the main app.
# All communication with the main thread should be handled carefully in one of these ways:
#     * the initial argument to this function...
#         * passed as copy
#         * shared through threading data structure (Lock, Event, etc)
#     * passed back and forth via send/yield
#         * need to send/yield copies of data
@thread_worker
def thread_work_loop_classic_detection(image, worker_alive):
    yield None # need to do this once just before setting worker to ready state
    worker_alive.set() # now this worker is ready to receive annotations
    image, is2d = make_3or4channel_image_grayscale(image)

    kwidth_opl = 4
    if is2d:
        precomputed_rpe_energy, precomputed_ilm_energy, img_bg_color = precompute_ilm_and_rpe_det_energy_maps(image)
        precomputed_opl_energy, precomputed_csj_energy = precompute_opl_csj_det_energy_maps(image, kwidth_opl, kwidth_csj)

    results = None
    while worker_alive.is_set():
        received = yield results # primary communication between this worker and the main thread
        if received is None:
            time.sleep(0.001) # this is needed so that this worker thread doesn't hog the CPU and freeze the main app
            results = None
            continue

        # receive updated user annotation data
        # from the main thread's "update_auto_detection_worker_send_user_annotations()"
        user_rpe_annos, unaligned_annos, rpe_align_annos, slice_3d, fovea_clicks, fovea_clicks_are_aligned2rpe, parafovea_width_px, hide_left_paraf, hide_right_paraf, det_max_slope = received
        user_rpe_annos = np.array(user_rpe_annos)
        unaligned_annos = np.array(unaligned_annos)
        rpe_align_annos = np.array(rpe_align_annos)
        det_max_slope = max(1, det_max_slope)

        if is2d:
            imgslice = image
        else:
            imgslice = image[slice_3d]
            precomputed_rpe_energy, precomputed_ilm_energy, img_bg_color = precompute_ilm_and_rpe_det_energy_maps(imgslice)
            precomputed_opl_energy, precomputed_csj_energy = precompute_opl_csj_det_energy_maps(imgslice, kwidth_opl, kwidth_csj)

        # detect RPE and ILM
        gauss_ilmrpe = 2.
        im_reg, ilm_rpe_raw, ilm_rpe_reg = new_ilmrpe_detect_rpe_only_first(imgslice, img_bg_color, precomputed_rpe_energy, precomputed_ilm_energy, user_rpe_annos, gauss_ilmrpe, gauss_ilmrpe, det_max_slope)

        unaligned_annos = aligned_to_unaligned_2d(unaligned_annos, rpe_align_annos, ilm_rpe_raw)
        iroc_raw = ridge_detect_opl_csj_given_user_annos(precomputed_opl_energy, precomputed_csj_energy, ilm_rpe_raw, unaligned_annos, rpe_thickness, det_max_slope)

        computed_annos, computed_reg_annos, metrics = calculate_metrics(imgslice.shape, iroc_raw, user_rpe_annos.dtype,
                                                            fovea_clicks, fovea_clicks_are_aligned2rpe, parafovea_width_px, hide_left_paraf, hide_right_paraf)

        # results will be returned to the main thread via yield
        results = (slice_3d, im_reg, computed_annos, computed_reg_annos, iroc_raw, metrics)





# This is run in a worker thread, separate from the main app.
# All communication with the main thread should be handled carefully in one of these ways:
#     * the initial argument to this function...
#         * passed as copy
#         * shared through threading data structure (Lock, Event, etc)
#     * passed back and forth via send/yield
#         * need to send/yield copies of data
@thread_worker
def thread_work_loop_neural_network(image, worker_alive):
    yield None # need to do this once just before setting worker to ready state
    worker_alive.set() # now this worker is ready to receive annotations
    import onnxruntime # need to install this: python3 -m pip install onnxruntime

    image, is2d = make_3or4channel_image_grayscale(image)

    session = onnxruntime.InferenceSession(onnx_weights_path, providers=['CPUExecutionProvider',])

    if is2d:
        assert len(image.shape) == 2, str(image.shape)
        precomputed_csj_energy, img_bg_color = precompute_csj_energy_map(image, kwidth_csj=kwidth_csj)
        medf = np.expand_dims(preprocess_image_for_onnx_inference(image), axis=0)
        (det_dnn,) = session.run(["output",], {"input":medf})
        det_dnn = (smooth_onnx_detections(det_dnn)*(-1.0)).squeeze(0).transpose(0,2,1)
        precomputed_opl_energy = det_dnn[4].transpose().astype(np.float64); precomputed_opl_energy -= precomputed_opl_energy.max()
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

        # receive updated user annotation data
        # from the main thread's "update_auto_detection_worker_send_user_annotations()"
        user_rpe_annos, unaligned_annos, rpe_align_annos, slice_3d, fovea_clicks, fovea_clicks_are_aligned2rpe, parafovea_width_px, hide_left_paraf, hide_right_paraf, det_max_slope = received
        user_rpe_annos = np.array(user_rpe_annos)
        unaligned_annos = np.array(unaligned_annos)
        rpe_align_annos = np.array(rpe_align_annos)
        det_max_slope = max(1, det_max_slope)

        unaligned_annos = np.maximum(unaligned_annos, user_rpe_annos) # merge

        if is2d:
            im_reg = image
        else:
            im_reg = image[slice_3d]
            precomputed_csj_energy, img_bg_color = precompute_csj_energy_map(im_reg, kwidth_csj=kwidth_csj)
            (det_dnn,) = session.run(["output",], {"input":medf[slice_3d:slice_3d+1]})
            det_dnn = (smooth_onnx_detections(det_dnn)*(-1.0)).squeeze(0).transpose(0,2,1)
            precomputed_opl_energy = det_dnn[4].transpose().astype(np.float64); precomputed_opl_energy -= precomputed_opl_energy.max()

        iroc_raw = masked_sequential_iroc_with_rioc_user_annos(det_dnn, unaligned_annos,
                                                               ilm_nms_w=15, rpe_top_mask=21, ilm_top_mask=4, max_slope=det_max_slope)
        iroc_raw = irgior_to_iroc(iroc_raw)

        # already detected the OPL, but detect it again in case user annotated in aligned mode, in which case "rpe_align_annos" has some annotations
        unaligned_annos = aligned_to_unaligned_2d(unaligned_annos, rpe_align_annos, iroc_raw)
        iroc_raw = ridge_detect_opl_csj_given_user_annos(precomputed_opl_energy, precomputed_csj_energy, iroc_raw, unaligned_annos, rpe_thickness, det_max_slope, opl_prior_two_thirds=False)


        im_reg, iroc_reg = straighten_rpe(im_reg, iroc_raw, img_bg_color)

        computed_annos, computed_reg_annos, metrics = calculate_metrics(medf.shape[1:], iroc_raw, user_rpe_annos.dtype,
                                                            fovea_clicks, fovea_clicks_are_aligned2rpe, parafovea_width_px, hide_left_paraf, hide_right_paraf)

        # results will be returned to the main thread via yield
        results = (slice_3d, im_reg, computed_annos, computed_reg_annos, iroc_raw, metrics)