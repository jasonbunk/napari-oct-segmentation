import numpy as np
from layer_group_alignment import aligned_to_unaligned_2d, points_align, unaligned_to_aligned_dets

from definitions_and_config import (
    RetinaLayers,
    metrics_table_rows,
    typical_image_width_pixels,
    typical_image_width_mm,
    typical_image_height_pixels,
    typical_image_height_mm,
    radius1_cal,
)


def calculate_metrics(
    im_shape,
    iroc_raw: np.ndarray,
    annos_dtype,
    fovea_clicks: np.ndarray | None,
    fovea_clicks_are_aligned2rpe: bool,
    parafovea_width_px: float,
    hide_left_paraf: bool,
    hide_right_paraf: bool
):
    """
    Arguments:
        im_shape
            2 integers for the shape of the input image, like (height, width)
        iroc_raw
            a 2D array of shape (width,4) where the 4 components start with I,R,O,C
            for each of I,R,O,C, for each image column it specifies the vertical location of the line
            I = ILM, R = RPE, O = OPL, C = CSJ
        annos_dtype
            the numerical data type of the annotations, like int32
        fovea_clicks
            might be None if nothing is clicked. Else it contains three 2D coordinates of the clicked fovea points.
            The clicks are allowed to be in any order; this function will sort them left-to-right
        fovea_clicks_are_aligned2rpe
            were the fovea clicks made in "RPE aligned" visualization mode?
        parafovea_width_px
            Radius in pixels around the fovea center where parafovea thickness is to be measured.
        hide_left_paraf
            Optional button prevents calculations from using left parafovea numbers (they will become nan)
        hide_right_paraf
            Optional button prevents calculations from using right parafovea numbers (they will become nan)
    """

    computed_annos = np.zeros(im_shape[-2:], dtype=annos_dtype)
    computed_reg_annos = np.zeros_like(computed_annos)

    # align detections to RPE
    iroc_reg = unaligned_to_aligned_dets(im_shape[-2]//2, iroc_raw)
    iroc_reg = np.maximum(0, np.minimum(iroc_reg, im_shape[-2]-1)) # clamp

    # Rename these to reduce confusion.
    # Each is an array of length #columns; where each number is the row at which the layer was detected in that column
    rowILM = iroc_reg[:,0] # I
    rowRPE = iroc_reg[:,1] # R
    rowOPL = iroc_reg[:,2] # O
    rowCSJ = iroc_reg[:,3] # C

    # not "anti-aliased", so if you zoom in, the line may look like disconnected dots
    cols = np.arange(computed_annos.shape[1])
    computed_annos[iroc_raw[:,0], cols] = RetinaLayers.ILM.value
    computed_annos[iroc_raw[:,1], cols] = RetinaLayers.RPE.value
    computed_reg_annos[rowILM, cols] = RetinaLayers.ILM.value
    computed_reg_annos[rowRPE, cols] = RetinaLayers.RPE.value
    computed_reg_annos[rowOPL, cols] = RetinaLayers.OPL.value
    computed_reg_annos[rowCSJ, cols] = RetinaLayers.CSJ.value

    # merge/align the above two
    computed_annos = aligned_to_unaligned_2d(computed_annos, computed_reg_annos, iroc_raw)

    metrics = [0 for _ in range(len(metrics_table_rows))]
    mr = {vv:ii for (ii,vv) in enumerate(metrics_table_rows)} # mr["x"] is easier to type than metrics_table_rows.index("x")

    if fovea_clicks is not None and len(fovea_clicks) == 3:
        # if there are 3 fovea clicks, use them to perform metrics calculations!

        if not fovea_clicks_are_aligned2rpe:
            # fovea angle must be measured as aligned to RPE
            fovea_clicks = points_align(fovea_clicks, iroc_raw, computed_annos.shape[-2]//2, u2a=True)

        clamp_col = lambda x: max(0, min(im_shape[1] - 1, int(round(x))))
        fovea_clicks = np.stack(sorted([fovea_clicks[cc] for cc in range(len(fovea_clicks))], key=lambda x: x[1]))
        fovea_center_x = fovea_clicks[1,1]
        parafovea_x1 = fovea_center_x - parafovea_width_px
        parafovea_x2 = fovea_center_x + parafovea_width_px

        fv_cen_left = clamp_col(fovea_center_x - radius1_cal)
        fv_cen_rigt = clamp_col(fovea_center_x + radius1_cal + 1)

        pfv_x1_left = clamp_col(parafovea_x1 - radius1_cal)
        pfv_x1_rigt = clamp_col(parafovea_x1 + radius1_cal + 1)

        pfv_x2_left = clamp_col(parafovea_x2 - radius1_cal)
        pfv_x2_rigt = clamp_col(parafovea_x2 + radius1_cal + 1)

        pix2mm = float(typical_image_height_mm / typical_image_height_pixels)

        # can choose np.mean or np.median
        stat = np.median

        metrics[mr["foveal_thickness_inner"]] = stat(rowOPL[fv_cen_left:fv_cen_rigt] - rowILM[fv_cen_left:fv_cen_rigt]) * pix2mm
        metrics[mr["foveal_thickness_outer"]] = stat(rowRPE[fv_cen_left:fv_cen_rigt] - rowOPL[fv_cen_left:fv_cen_rigt]) * pix2mm
        metrics[mr["foveal_thickness"]] = metrics[mr["foveal_thickness_inner"]] + metrics[mr["foveal_thickness_outer"]]

        pti_left = stat(rowOPL[pfv_x1_left:pfv_x1_rigt] - rowILM[pfv_x1_left:pfv_x1_rigt]) * pix2mm
        pti_rigt = stat(rowOPL[pfv_x2_left:pfv_x2_rigt] - rowILM[pfv_x2_left:pfv_x2_rigt]) * pix2mm

        pto_left = stat(rowRPE[pfv_x1_left:pfv_x1_rigt] - rowOPL[pfv_x1_left:pfv_x1_rigt]) * pix2mm
        pto_rigt = stat(rowRPE[pfv_x2_left:pfv_x2_rigt] - rowOPL[pfv_x2_left:pfv_x2_rigt]) * pix2mm

        if hide_left_paraf:
            pti_left = float('nan')
            pto_left = float('nan')
        if hide_right_paraf:
            pti_rigt = float('nan')
            pto_rigt = float('nan')

        metrics[mr["parafoveal_thickness_inner_left"]] = pti_left
        metrics[mr["parafoveal_thickness_inner_right"]] = pti_rigt
        metrics[mr["parafoveal_thickness_inner"]] = (pti_left + pti_rigt) / 2.

        metrics[mr["parafoveal_thickness_outer_left"]] = pto_left
        metrics[mr["parafoveal_thickness_outer_right"]] = pto_rigt
        metrics[mr["parafoveal_thickness_outer"]] = (pto_left + pto_rigt) / 2.

        fovleft = fovea_clicks[0] - fovea_clicks[1] # vector pointing from fovea center to left edge
        fovrigt = fovea_clicks[2] - fovea_clicks[1] # vector pointing from fovea center to right edge
        metrics[mr["foveal_angle"]] = np.arccos(np.dot(fovleft, fovrigt) / (np.linalg.norm(fovleft) * np.linalg.norm(fovrigt))) * 180 / np.pi

        # consider recalculating foveal angle based on aspect ratio of pixels??

        metrics[mr["choroid_foveal_thickness"]] = stat(rowCSJ[fv_cen_left:fv_cen_rigt] - rowRPE[fv_cen_left:fv_cen_rigt]) * pix2mm

        cpt_left = stat(rowCSJ[pfv_x1_left:pfv_x1_rigt] - rowRPE[pfv_x1_left:pfv_x1_rigt]) * pix2mm
        cpt_rigt = stat(rowCSJ[pfv_x2_left:pfv_x2_rigt] - rowRPE[pfv_x2_left:pfv_x2_rigt]) * pix2mm
        if hide_left_paraf:
            cpt_left = float('nan')
        if hide_right_paraf:
            cpt_rigt = float('nan')
        metrics[mr["choroid_parafoveal_thickness_left"]] = cpt_left
        metrics[mr["choroid_parafoveal_thickness_right"]] = cpt_rigt
        metrics[mr["choroid_parafoveal_thickness"]] = (cpt_left + cpt_rigt) / 2.

        for lrn in ("", "_left", "_right"):
            metrics[mr["f_p_ratio_inner"+lrn]] = metrics[mr["foveal_thickness_inner"]] / metrics[mr["parafoveal_thickness_inner"+lrn]]
            metrics[mr["f_p_ratio_outer"+lrn]] = metrics[mr["foveal_thickness_outer"]] / metrics[mr["parafoveal_thickness_outer"+lrn]]

    # formatting
    metrics = list([ [value,] for value in metrics ])

    return computed_annos, computed_reg_annos, metrics
