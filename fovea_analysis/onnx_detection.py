# Functions for pre-processing and post-processing around ONNX DNN model
# Post-processing traces lines using dynamic programming (seam carving) while incorporating user corrections
# author: Jason Bunk
import os
import numpy as np
from skimage.filters import median as median_filter, gaussian as gaussian_filter
from seamsloped import get_sloped_seam
from numba import njit
from enum import IntEnum

# These are the layers the neural network was trained to detect
class RetinaLayersIRGIOR(IntEnum):
    ILM = 1
    RNFLGCL = 2
    GCLIPL = 3
    IPLINL = 4
    OPL = 5
    RPE = 6

def irgior_to_iroc(irgior:np.ndarray):
    assert len(irgior.shape) == 2 and int(irgior.shape[1]) == 6, str(irgior.shape)
    iroc = np.zeros((irgior.shape[0], 4), dtype=irgior.dtype)
    iroc[:,0] = irgior[:,0]
    iroc[:,1] = irgior[:,5]
    iroc[:,2] = irgior[:,4]
    return iroc

thispath = os.path.dirname(os.path.abspath(__file__))
onnx_weights_path = os.path.join(thispath, "dnn_weights_dataseed1144716146_rnn1d.onnx")
assert os.path.isfile(onnx_weights_path), onnx_weights_path


def preprocess_image_for_onnx_inference(image:np.ndarray):
    assert len(image.shape) == 2, str(image.shape)
    assert image.dtype == np.uint8, str(image.dtype)

    medf = median_filter(image, np.ones((5,5),dtype=np.uint8))
    #import cv2; medf = cv2.medianBlur(image, 5)

    medtosubtr = float(np.median(medf))

    medf = np.stack((
        (image.astype(np.float32) - medtosubtr)/25.5,
        (medf.astype(np.float32) - medtosubtr)/25.5,
    ))
    return medf


@njit
def make_mask_for_rpe_det(rpe_anno, top_mask_w, value_of_nonzero:float):
    """
    for each image row:
        if user made an annotation, respect that: use mask to enforce constraint that detection should occur on user-annotated pixels
        otherwise, detection can be anywhere
    """
    mask = np.zeros(rpe_anno.shape, dtype=np.float64)
    for ii in range(mask.shape[0]):
        if np.any(rpe_anno[ii,:]): # if any pixel in this row was annotated
            mask[ii,:] = rpe_anno[ii,:].astype(np.float64) * value_of_nonzero * 5. # set mask row to user annotation, so that detection should occur within masked pixels
        else:
            # if user gives no hint, look for RPE anywhere except near the edge of the image
            mask[ii,top_mask_w:] = value_of_nonzero
    return mask


@njit
def make_mask_for_det_left_of_rpe(annos2d, rpe_det, which_layer_value, top_mask_w, rpe_width, value_of_nonzero:float):
    """
    for each image row:
        if user made an annotation, respect that: use mask to enforce constraint that detection should occur on user-annotated pixels
        otherwise, use mask to enforce constraint that the layer needs to be detected *above* the RPE
    """
    mask = np.zeros(annos2d.shape, dtype=np.float64)
    for ii in range(mask.shape[0]):
        if np.any(annos2d[ii,:]): # if any pixel in this row was annotated
            mask[ii,:] = annos2d[ii,:].astype(np.float64) * value_of_nonzero * 5. # set mask row to user annotation, so that detection should occur within masked pixels
        else:
            # if user gives no hint, look for ILM somewhere above RPE (i.e. to the left of it, since this image is transposed)
            mask[ii, top_mask_w:max(top_mask_w+1,rpe_det[ii]-rpe_width)] = value_of_nonzero
    return mask


@njit
def make_mask_for_det_in_between_two_layers(anno2d, above_this, above_pad, below_this, below_pad, value_of_nonzero:float):
    """
    for each image row:
        if user made an annotation, respect that: use mask to enforce constraint that detection should occur on user-annotated pixels
        otherwise, use mask to enforce constraint that the layer needs to be detected in between specified layers
    """
    mask = np.zeros(anno2d.shape, dtype=np.float64)
    last = anno2d.shape[1] - 1
    for ii in range(mask.shape[0]):
        if np.any(anno2d[ii,:]): # if any pixel in this row was annotated
            mask[ii,:] = anno2d[ii,:].astype(np.float64) * value_of_nonzero * 5. # set mask row to user annotation, so that detection should occur within masked pixels
        else:
            # if user gives no hint, look for ILM somewhere above RPE (i.e. to the left of it, since this image is transposed)
            above = max(1, above_this[ii] - above_pad)
            below = min(last, below_this[ii] + below_pad)
            if above <= below:
                mask[ii, int(round((above+below)/2.))] = value_of_nonzero
            else:
                mask[ii, below:above+1] = value_of_nonzero
    return mask


#@njit
def _compute_masked_irx_detection_with_user_annos(preds:np.ndarray, userannos:np.ndarray, detect_these_layers:tuple, ilm_nms_w:int, ilm_top_mask:int, rpe_top_mask:int, max_slope:int):
    ret = np.empty((preds.shape[1], preds.shape[0]), dtype=np.int64)
    #mfkern = np.ones((3,), dtype=np.uint8)
    smooth_det = lambda x_: x_ #median_filter(x_, mfkern)
    rpe_idx = RetinaLayersIRGIOR.RPE.value - 1
    ilm_idx = RetinaLayersIRGIOR.ILM.value - 1
    ret[:,rpe_idx] = smooth_det(get_sloped_seam(preds[rpe_idx,:,:].astype(np.float64) - make_mask_for_rpe_det(np.equal(userannos,1+rpe_idx), rpe_top_mask, 1e5), max_slope))
    ret[:,ilm_idx] = smooth_det(get_sloped_seam(preds[ilm_idx,:,:].astype(np.float64) - make_mask_for_det_left_of_rpe(np.equal(userannos,1+ilm_idx), ret[:,rpe_idx], 1+ilm_idx, ilm_top_mask, ilm_nms_w, 1e5), max_slope))
    for ii in detect_these_layers:
        if ii != rpe_idx and ii != ilm_idx:
            ret[:,ii] = smooth_det(get_sloped_seam(preds[ii,:,:].astype(np.float64) - make_mask_for_det_in_between_two_layers(np.equal(userannos, 1+ii), ret[:,rpe_idx], 1, ret[:,ilm_idx], 1, 1e5), max_slope))
    return ret

def masked_irx_detection_with_user_annos(preds:np.ndarray, userannos:np.ndarray, detect_these_layers:tuple, ilm_nms_w:int, ilm_top_mask:int, rpe_top_mask:int, max_slope:int):
    assert len(preds.shape) == 3, str(preds.shape)
    assert len(userannos.shape) == 2, str(userannos.shape)
    assert int(preds.shape[1]) == int(userannos.shape[1]), f"{preds.shape} vs {userannos.shape}"
    assert int(preds.shape[2]) == int(userannos.shape[0]), f"{preds.shape} vs {userannos.shape}"
    assert int(preds.shape[0]) == 6, str(preds.shape) # expect irgior, see definitions_and_config and weights above
    return _compute_masked_irx_detection_with_user_annos(preds, userannos.transpose(), detect_these_layers, ilm_nms_w, ilm_top_mask, rpe_top_mask, max_slope)


#@njit
def _compute_masked_sequential_detection_with_user_annos(preds:np.ndarray, userannos:np.ndarray, ilm_nms_w:int, rpe_top_mask:int, ilm_top_mask:int, max_slope:int):
    ret = np.empty((preds.shape[1], preds.shape[0]), dtype=np.int64)
    #mfkern = np.ones((3,), dtype=np.uint8)
    smooth_det = lambda x_: x_ #median_filter(x_, mfkern)
    rpe_idx = RetinaLayersIRGIOR.RPE.value - 1
    ilm_idx = RetinaLayersIRGIOR.ILM.value - 1
    ipn_idx = RetinaLayersIRGIOR.IPLINL.value - 1
    gci_idx = RetinaLayersIRGIOR.GCLIPL.value - 1
    rng_idx = RetinaLayersIRGIOR.RNFLGCL.value - 1
    opl_idx = RetinaLayersIRGIOR.OPL.value - 1
    ret[:,rpe_idx] = smooth_det(get_sloped_seam(preds[rpe_idx,:,:].astype(np.float64) - make_mask_for_rpe_det(np.equal(userannos,1+rpe_idx), rpe_top_mask, 1e5), max_slope))
    ret[:,ilm_idx] = smooth_det(get_sloped_seam(preds[ilm_idx,:,:].astype(np.float64) - make_mask_for_det_left_of_rpe(np.equal(userannos,1+ilm_idx), ret[:,rpe_idx], 1+ilm_idx, ilm_top_mask, ilm_nms_w, 1e5), max_slope))
    ret[:,ipn_idx] = smooth_det(get_sloped_seam(preds[ipn_idx,:,:].astype(np.float64) - make_mask_for_det_in_between_two_layers(np.equal(userannos, 1+ipn_idx), ret[:,rpe_idx], 1, ret[:,ilm_idx], 1, 1e5), max_slope))
    ret[:,gci_idx] = smooth_det(get_sloped_seam(preds[gci_idx,:,:].astype(np.float64) - make_mask_for_det_in_between_two_layers(np.equal(userannos, 1+gci_idx), ret[:,ipn_idx], 1, ret[:,ilm_idx], 1, 1e5), max_slope))
    ret[:,rng_idx] = smooth_det(get_sloped_seam(preds[rng_idx,:,:].astype(np.float64) - make_mask_for_det_in_between_two_layers(np.equal(userannos, 1+rng_idx), ret[:,gci_idx], 1, ret[:,ilm_idx], 1, 1e5), max_slope))
    ret[:,opl_idx] = smooth_det(get_sloped_seam(preds[opl_idx,:,:].astype(np.float64) - make_mask_for_det_in_between_two_layers(np.equal(userannos, 1+opl_idx), ret[:,rpe_idx], 1, ret[:,ilm_idx], 1, 1e5), max_slope))
    return ret

def masked_sequential_detection_with_user_annos(preds:np.ndarray, userannos:np.ndarray, ilm_nms_w:int, rpe_top_mask:int, ilm_top_mask:int, max_slope:int):
    assert len(preds.shape) == 3, str(preds.shape)
    assert len(userannos.shape) == 2, str(userannos.shape)
    assert int(preds.shape[1]) == int(userannos.shape[1]), f"{preds.shape} vs {userannos.shape}"
    assert int(preds.shape[2]) == int(userannos.shape[0]), f"{preds.shape} vs {userannos.shape}"
    assert int(preds.shape[0]) == 6, str(preds.shape) # expect irgior, see definitions_and_config and weights above
    return _compute_masked_sequential_detection_with_user_annos(preds, userannos.transpose(), ilm_nms_w, rpe_top_mask, ilm_top_mask, max_slope)

def smooth_onnx_detections(preds:np.ndarray):
    assert len(preds.shape) == 4, str(preds.shape)
    assert int(preds.shape[1]) == 6, str(preds.shape)
    return np.stack([np.stack([gaussian_filter(preds[ii,jj], sigma=3., mode='reflect') for jj in range(preds.shape[1])]) for ii in range(preds.shape[0])])


# this is different from above because instead of user annotations being irgoir, user annotations are rioc... makes things more confusing to mix schemas
#@njit
def _compute_masked_sequential_iroc_with_rioc_user_annos(preds:np.ndarray, userannos:np.ndarray, ilm_nms_w:int, rpe_top_mask:int, ilm_top_mask:int, max_slope:int):
    ret = np.empty((preds.shape[1], preds.shape[0]), dtype=np.int64)
    #mfkern = np.ones((3,), dtype=np.uint8)
    smooth_det = lambda x_: x_ #median_filter(x_, mfkern)
    rpe_idx = RetinaLayersIRGIOR.RPE.value - 1
    ilm_idx = RetinaLayersIRGIOR.ILM.value - 1
    opl_idx = RetinaLayersIRGIOR.OPL.value - 1
    ret[:,rpe_idx] = smooth_det(get_sloped_seam(preds[rpe_idx,:,:].astype(np.float64) - make_mask_for_rpe_det(np.equal(userannos,1), rpe_top_mask, 1e5), max_slope))
    ret[:,ilm_idx] = smooth_det(get_sloped_seam(preds[ilm_idx,:,:].astype(np.float64) - make_mask_for_det_left_of_rpe(np.equal(userannos,2), ret[:,rpe_idx], 1+ilm_idx, ilm_top_mask, ilm_nms_w, 1e5), max_slope))
    ret[:,opl_idx] = smooth_det(get_sloped_seam(preds[opl_idx,:,:].astype(np.float64) - make_mask_for_det_in_between_two_layers(np.equal(userannos,3), ret[:,rpe_idx], 1, ret[:,ilm_idx], 1, 1e5), max_slope))
    return ret

def masked_sequential_iroc_with_rioc_user_annos(preds:np.ndarray, userannos:np.ndarray, ilm_nms_w:int, rpe_top_mask:int, ilm_top_mask:int, max_slope:int):
    assert len(preds.shape) == 3, str(preds.shape)
    assert len(userannos.shape) == 2, str(userannos.shape)
    assert int(preds.shape[1]) == int(userannos.shape[1]), f"{preds.shape} vs {userannos.shape}"
    assert int(preds.shape[2]) == int(userannos.shape[0]), f"{preds.shape} vs {userannos.shape}"
    assert int(preds.shape[0]) == 6, str(preds.shape) # expect irgior, see definitions_and_config and weights above
    return _compute_masked_sequential_iroc_with_rioc_user_annos(preds, userannos.transpose(), ilm_nms_w, rpe_top_mask, ilm_top_mask, max_slope)
