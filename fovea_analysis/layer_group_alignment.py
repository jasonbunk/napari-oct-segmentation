# Author: Jason Bunk
import napari
import numpy as np
from numba import njit
import globals


class LayersGroup:
    def __init__(self):
        self.image_layer = None
        self.user_labels = None
        self.auto_labels = None


@njit
def merge_labels(aa, bb):
    return np.maximum(aa, bb)


@njit
def aligned_to_unaligned_dets(alignto_row_idx, regloc, rawloc):
    unaloc = np.copy(regloc)
    for jj in range(regloc.shape[0]):
        unaloc[jj,:] += rawloc[jj,1] - alignto_row_idx
    return unaloc


@njit
def unaligned_to_aligned_dets(alignto_row_idx, rawloc):
    regloc = np.copy(rawloc)
    for jj in range(regloc.shape[0]):
        regloc[jj,:] -= rawloc[jj,1] - alignto_row_idx
    return regloc


@njit
def unaligned_to_aligned_2d(unaligned, aligned, dets):
    regalignto = aligned.shape[-2]//2
    for jj in range(dets.shape[0]):
        if dets[jj,1] == regalignto:
            aligned[:,jj] = merge_labels(aligned[:,jj], unaligned[:,jj])
        elif dets[jj,1] < regalignto:
            aligned[(regalignto-dets[jj,1]):, jj] = merge_labels(aligned[(regalignto-dets[jj,1]):, jj], unaligned[:-(regalignto-dets[jj,1]),jj])
        else:
            aligned[:-(dets[jj,1]-regalignto), jj] = merge_labels(aligned[:-(dets[jj,1]-regalignto), jj], unaligned[(dets[jj,1]-regalignto):,jj])
    return aligned


@njit
def aligned_to_unaligned_2d(unaligned, aligned, dets):
    regalignto = aligned.shape[-2]//2
    for jj in range(dets.shape[0]):
        if dets[jj,1] == regalignto:
            unaligned[:,jj] = merge_labels(aligned[:,jj], unaligned[:,jj])
        elif dets[jj,1] < regalignto:
            unaligned[:-(regalignto-dets[jj,1]), jj] = merge_labels(aligned[(regalignto-dets[jj,1]):, jj], unaligned[:-(regalignto-dets[jj,1]),jj])
        else:
            unaligned[(dets[jj,1]-regalignto):, jj] = merge_labels(aligned[:-(dets[jj,1]-regalignto), jj], unaligned[(dets[jj,1]-regalignto):,jj])
    return unaligned


@njit
def points_align(points, dets, regmiddle, u2a:bool):
    for ii in range(len(points)):
        jj = np.int64(np.round(points[ii,1])) # points are (row,col)
        jj = max(0, min(dets.shape[0]-1, jj)) # clamp to valid image column
        if u2a:
            points[ii,0] -= dets[jj,1] - regmiddle
        else:
            points[ii,0] += dets[jj,1] - regmiddle
    return points


# TODO these should fully copy 3D, not just current slice

def toggle_viz_unaligned_to_aligned(unaligned, aligned, user_points):
    napari.current_viewer().layers.selection.select_only(aligned.user_labels)
    if globals.latest_detected_layers is not None:
        latest_det = np.copy(globals.latest_detected_layers)
        if len(unaligned.user_labels.data.shape) == 2:
            aligned.user_labels.data = unaligned_to_aligned_2d(np.copy(unaligned.user_labels.data), np.copy(aligned.user_labels.data), latest_det)
        else:
            slice3d = napari.current_viewer().dims.current_step[0]
            copyof = np.copy(aligned.user_labels.data)
            copyof[slice3d] = unaligned_to_aligned_2d(np.copy(unaligned.user_labels.data[slice3d]), copyof[slice3d], latest_det)
            aligned.user_labels.data = copyof
        if len(user_points.data) > 0:
            user_points.data = points_align(np.array(user_points.data), latest_det, aligned.user_labels.data.shape[-2] // 2, u2a=True)


def toggle_viz_aligned_to_unaligned(unaligned, aligned, user_points):
    napari.current_viewer().layers.selection.select_only(unaligned.user_labels)
    if globals.latest_detected_layers is not None:
        latest_det = np.copy(globals.latest_detected_layers)
        if len(unaligned.user_labels.data.shape) == 2:
            unaligned.user_labels.data = aligned_to_unaligned_2d(np.copy(unaligned.user_labels.data), np.copy(aligned.user_labels.data), latest_det)
        else:
            slice3d = napari.current_viewer().dims.current_step[0]
            copyof = np.copy(unaligned.user_labels.data)
            copyof[slice3d] = aligned_to_unaligned_2d(copyof[slice3d], np.copy(aligned.user_labels.data[slice3d]), latest_det)
            unaligned.user_labels.data = copyof
        if len(user_points.data) > 0:
            user_points.data = points_align(np.array(user_points.data), latest_det, aligned.user_labels.data.shape[-2] // 2, u2a=False)


if __name__ == "__main__":
    # test: aligned_to_unaligned(unaligned_to_aligned()) == identity
    import math
    import matplotlib.pyplot as plt

    dummy_annos = np.zeros((100,100), dtype=np.uint8)
    dummy_annos[12:15, 30:60] = 255
    alignment = np.zeros((dummy_annos.shape[1], 4), dtype=np.int64)

    for ii in range(dummy_annos.shape[1]):
        alignment[ii,1] = int(round(50.0 + 20.0 * math.sin(float(ii)/10.)))

    warped = unaligned_to_aligned_2d(dummy_annos, np.zeros_like(dummy_annos), alignment)
    backto = aligned_to_unaligned_2d(np.zeros_like(dummy_annos), warped, alignment)

    plt.figure(); plt.plot(alignment[:,1]); plt.grid(True)
    plt.figure(); plt.imshow(dummy_annos); plt.title('original')
    plt.figure(); plt.imshow(warped); plt.title('warped')
    plt.figure(); plt.imshow(backto); plt.title('should be same as original')
    plt.show()

    assert np.amax(np.abs(backto - dummy_annos)) == 0
