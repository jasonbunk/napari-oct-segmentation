# Author: Jason Bunk
import math
import numpy as np
from numba import njit
#import cv2
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import correlate_sparse, gaussian
from skimage.morphology import dilation
from my_utils import uint8clip, uint8norm
from definitions_and_config import RetinaLayers, RetinaLayerColors
from seamsloped import get_sloped_seam


@njit
def straighten_rpe(img2d, regloc, fill_color):
    """ in each image column, shift up or down such that RPE becomes in the middle """
    registered = np.full(img2d.shape, fill_value=fill_color, dtype=np.uint8)
    regalignto = img2d.shape[-2]//2
    adjloc = np.copy(regloc)
    adjloc[:,1] = regalignto
    for jj in range(regloc.shape[0]):
        adjloc[jj,0] -= regloc[jj,1] - regalignto
        if regloc[jj,1] == regalignto:
            registered[:,jj] = img2d[:,jj]
        elif regloc[jj,1] < regalignto:
            registered[(regalignto-regloc[jj,1]):, jj] = img2d[:-(regalignto-regloc[jj,1]),jj]
        else:
            registered[:-(regloc[jj,1]-regalignto), jj] = img2d[(regloc[jj,1]-regalignto):,jj]
    return registered, adjloc


@njit
def build_edge_detection_kernel(epixels:int):
    """ one-dimensional edge detection filter kernel """
    assert epixels % 2 == 1, str(epixels)
    kw = (epixels-1)//2
    kernel = np.exp(-np.abs(np.linspace(-kw,kw,epixels))/(kw/4.5))
    kernel[kw] = 0.
    kernel /= kernel[:kw].sum()
    kernel[kw+1:] *= -1.
    return kernel.astype(np.float32)


def positive_ridge_filter(float_image2d, filter_width, boundary_fill_color):
    """ ridge detection, for a high ridge surrounded by low values """
    padbnd = 9+int(round(filter_width*3.))
    result = np.pad(float_image2d, padbnd, mode='linear_ramp', end_values=boundary_fill_color) # smooth boundary condition
    _, result = hessian_matrix_eigvals(hessian_matrix(result, filter_width)) # ridge detection
    result = result[padbnd:-padbnd, padbnd:-padbnd]
    result[result>0] = 0
    return result


@njit
def fill_cols_with_no_user_annos(array2d, amaxpercol):
    for jj in range(len(amaxpercol)):
        if amaxpercol[jj] < 1e-6:
            array2d[:,jj] = 1.
    return array2d
def create_smoothedpadded_mask_for_user_anno_label(annos2d, label, pad_amt:int):
    assert len(annos2d.shape) == 2, str(annos2d.shape)
    blurred = np.uint8(np.equal(annos2d, label)) # allocate what will be blurred
    if pad_amt > 1: # dilate AND blur, so like a fuzzy dilation, to slightly widen user annotation
        dkern = np.ones((pad_amt,1),dtype=np.uint8)
        blurred = gaussian(np.float32(dilation(blurred, dkern)), sigma=(0.3*((pad_amt-1)*0.5 - 1) + 0.8, 0), mode='reflect')
    return fill_cols_with_no_user_annos(np.copy(blurred), np.amax(blurred, axis=0)), blurred


@njit
def make_mask_for_ilm_det(annos2d, rpe_det, rpe_width):
    """
    for each image column:
        if user made an annotation, respect that: use mask to enforce constraint that detection should occur on user-annotated pixels
        otherwise, use mask to enforce constraint that ILM needs to be detected *above* the RPE
    """
    ilm_anno = np.equal(annos2d, RetinaLayers.ILM.value)
    mask = np.zeros(ilm_anno.shape, dtype=np.float32)
    usrm = np.zeros_like(mask)
    for jj in range(ilm_anno.shape[1]):
        if np.any(ilm_anno[:,jj]): # if any pixel in this column was annotated
            mask[:,jj] = ilm_anno[:,jj].astype(np.uint8) # set mask column to user annotation, so that detection should occur within masked pixels
            usrm[:,jj] = mask[:,jj] # also save another copy of user annotations
        else:
            # if user gives no hint, look for ILM somewhere above RPE
            mask[:max(1,rpe_det[jj]-rpe_width), jj] = 1.
    return mask, usrm


def precompute_ilm_and_rpe_det_energy_maps(img2d:np.ndarray):
    """
        Algorithm-suggesting energy maps can be computed before any user annotations are made 
        This greatly speeds up time to compute updates after annotations are made
    """
    assert len(img2d.shape) == 2, str(img2d.shape)
    assert img2d.dtype == np.uint8, str(img2d.dtype)

    medcolor = float(np.median(img2d))
    imflt = np.float32(img2d) - medcolor
    bg_color = int(round(medcolor))

    # kernel is shaped sort of like a Gaussian... seems good at finding the RPE
    kernel = np.float32([0.20494800806045532, 0.21680282056331635, 0.0175921767950058, 0.21016886830329895, 0.06142314523458481, -0.13058465719223022, -0.09653705358505249, -0.23860086500644684, 0.0638413354754448, -0.43158215284347534, -0.27492719888687134, -0.6807354688644409, -0.873059868812561, -0.8838492035865784, -1.1232166290283203, -1.330405831336975, -1.4647537469863892, -1.5212706327438354, -1.3597962856292725, -1.2156505584716797, -1.022175908088684, -0.5830516219139099, 0.07981111854314804, 1.0377705097198486, 1.5698175430297852, 2.185596227645874, 3.1620688438415527, 5.0284013748168945, 5.9125285148620605, 6.68724250793457, 6.632073879241943, 6.456590175628662, 4.759913921356201, 3.9171087741851807, 2.583303689956665, 1.5451005697250366, 1.5879244804382324, 1.2084832191467285, 0.9176650047302246, 0.7239735126495361, 0.593952476978302, 0.23524045944213867, 0.0034794583916664124, -0.1426706612110138, -0.3108716905117035, -0.5295586585998535, -0.8073593378067017, -0.7612517476081848, -1.037959098815918, -1.0109935998916626, -0.9185844659805298, -0.967619776725769, -1.0670076608657837, -1.0641627311706543, -1.1542632579803467, -0.9417900443077087, -1.2053650617599487, -1.139441728591919, -1.0282217264175415, -0.8915272355079651, -1.2369379997253418]).reshape((-1,1))
    precomputed_rpe = correlate_sparse(imflt, kernel) #, mode='nearest')

    # edge detection kernel is good at finding the upper edge of the ILM, when the constraint is that it must be above the RPE
    kernel = build_edge_detection_kernel(kernel.size).reshape((-1,1))
    precomputed_ilm = correlate_sparse(imflt, kernel) #, mode='nearest')

    return precomputed_rpe, precomputed_ilm, bg_color


def new_ilmrpe_detect_rpe_only_first(img2d, bg_color, precomputed_rpe, precomputed_ilm, user_annos, gauss_rpe:float, gauss_ilm:float, detection_max_slope:int):
    """
        first, detects RPE... second, detects ILM...
        each uses "sloped seam carving" which is dynamic programming for optimal constrained line detection
            the detected line is optimal (traces minima the best) while obeying the constraint that it's a smooth line with a bounded slope
            minima are low-valued pixels in edge or ridge detection image...
                detection images are masked to enforce constraints, such as obeying user scribbles, or the constraint that the ILM is above the RPE
    """
    assert len(img2d.shape) == 2, str(img2d.shape)
    assert img2d.dtype == np.uint8, str(img2d.dtype)
    assert len(user_annos.shape) == 2, str(user_annos.shape)
    irlocs = np.empty((img2d.shape[1], 2), dtype=np.int64)

    rpe_width_nms = 19
    hacky_ilm_offset_hardcoded = 3

    det_mask, user_mask = create_smoothedpadded_mask_for_user_anno_label(user_annos, RetinaLayers.RPE.value, 5)
    detf = ((gaussian(precomputed_rpe * det_mask, sigma=gauss_rpe, mode='reflect') + 0.1) * det_mask).astype(np.float64) + (user_mask.astype(np.float64) * 1e5)
    irlocs[:,1] = get_sloped_seam(-1.0*detf.transpose(), detection_max_slope) # transpose image because we want to detect horizontal lines, but the seam code only detects vertical lines

    det_mask, user_mask = make_mask_for_ilm_det(np.pad(user_annos, ((0,hacky_ilm_offset_hardcoded),(0,0))), irlocs[:,1], rpe_width_nms)
    detf = ((gaussian(np.pad(precomputed_ilm,((hacky_ilm_offset_hardcoded,0),(0,0))) * det_mask, sigma=gauss_ilm, mode='reflect') - 0.1) * det_mask).astype(np.float64) - (user_mask.astype(np.float64) * 1e5)
    irlocs[:,0] = get_sloped_seam(detf.transpose(), detection_max_slope)

    registered, adjlocs = straighten_rpe(img2d, irlocs, bg_color)
    return registered, irlocs, adjlocs


@njit
def make_mask_for_opl_det(opl_anno, opl_col_anno, irlocs, prior_two_thirds:bool):
    mask = np.zeros(opl_anno.shape, dtype=np.float64)
    usrm = np.zeros_like(mask)
    for jj in range(opl_anno.shape[1]):
        if opl_col_anno[jj]:
            mask[:,jj] = opl_anno[:,jj]
            usrm[:,jj] = opl_anno[:,jj]
        else:
            # if user gives no hint, look for OPL in between ILM and RPE
            ilm = irlocs[jj,:2].min()
            rpe = irlocs[jj,:2].max()
            i2r = (rpe - ilm)
            if i2r <= 5 or not prior_two_thirds:
                mask[ilm:max(ilm+1,rpe),jj] = 1. # in this situation, ILM-RPE distance is too skinny for the below math to make sense
            else:
                startp = math.floor(float(ilm)+float(i2r)*0.1)
                endpt = math.ceil(float(ilm)+float(i2r)*0.9)
                midpt = int(round(float(ilm)+float(i2r)*0.667)) # the OPL tends to be roughly 2/3 of the way down from the ILM to the RPE
                for kk in range(startp,endpt):
                    # fuzz above and below guessed "midpt" because the OPL isn't always at the same position between ILM and RPE
                    if kk <= midpt:
                        mask[kk,jj] = np.square(float(kk-startp+1)/float(midpt-startp+1))
                    else:
                        mask[kk,jj] = np.square(1.0 - float(kk-midpt)/float(endpt-midpt))
    return mask, usrm


@njit
def make_mask_for_csj_det(csj_anno, csj_col_anno, irlocs, rpe_width):
    """
    for each image column:
        if user made an annotation, respect that: use mask to enforce constraint that detection should occur on user-annotated pixels
        otherwise, use mask to enforce constraint that CSJ needs to be detected *below* the RPE
    """
    mask = np.zeros(csj_anno.shape, dtype=np.float64)
    usrm = np.zeros_like(mask)
    for jj in range(csj_anno.shape[1]):
        if csj_col_anno[jj]: # if user annotated a pixel in this column, use mask to enforce that annotation
            mask[:,jj] = csj_anno[:,jj]
            usrm[:,jj] = csj_anno[:,jj]
        else:
            # if user gives no hint, look for CSJ somewhere below RPE
            mask[irlocs[jj,:2].max()+rpe_width:,jj] = 1.
    return mask, usrm


def precompute_opl_csj_det_energy_maps(img2d:np.ndarray, kwidth_opl:float, kwidth_csj:float):
    """
        Algorithm-suggesting energy maps can be computed before any user annotations are made 
        This greatly speeds up time to compute updates after annotations are made
    """
    assert len(img2d.shape) == 2, str(img2d.shape)
    assert img2d.dtype == np.uint8, str(img2d.dtype)

    opl_fil = np.float32(img2d)/255.
    csj_fil = np.copy(opl_fil)
    medcolor = np.median(opl_fil)
    # OPL detection
    opl_fil = positive_ridge_filter(opl_fil, kwidth_opl, medcolor).astype(np.float64)
    # CSJ = C/S Junction detection
    csj_fil = positive_ridge_filter(csj_fil, kwidth_csj, medcolor).astype(np.float64)

    return opl_fil, csj_fil


def precompute_csj_energy_map(img2d:np.ndarray, kwidth_csj:float):
    csj_fil = np.float32(img2d)/255.
    medcolor = np.median(csj_fil)
    return positive_ridge_filter(csj_fil, kwidth_csj, medcolor).astype(np.float64), medcolor*255.


def ridge_detect_opl_csj_given_user_annos(precomputed_opl, precomputed_csj, irlocs, user_annos_2d, rpe_width:int, detection_max_slope:int, visualize:bool=False, opl_prior_two_thirds:bool=True):
    """
        detects OPL, and then CSJ
        each uses "sloped seam carving" which is dynamic programming for optimal constrained line detection
            the detected line is optimal (traces minima the best) while obeying the constraint that it's a smooth line with a bounded slope
            minima are low-valued pixels in ridge detection image...
                detection images are masked to enforce constraints, such as obeying user scribbles, or the constraint that the CSJ is below the RPE
    """
    assert len(precomputed_opl.shape) == 2, f"opl {precomputed_opl.shape}"
    assert len(precomputed_csj.shape) == 2, f"csj {precomputed_csj.shape}"
    assert len(user_annos_2d.shape) == 2, f"user_annos_2d {user_annos_2d.shape}"

    iroc = np.pad(irlocs, ((0,0),(0,2)))

    # incorporate user annotation, else we only know that it should be between ILM and RPE
    opl_anno = np.equal(user_annos_2d, RetinaLayers.OPL.value)
    percolreducedanno = opl_anno.any(axis=0)
    opl_anno = opl_anno.astype(np.float64)
    opl_anno, usronly = make_mask_for_opl_det(opl_anno, percolreducedanno, irlocs, prior_two_thirds=opl_prior_two_thirds)
    opl_fil = (precomputed_opl - 1e-7) * opl_anno - (usronly * 5.)
    #iroc[:,2] = np.argmin(opl_fil, axis=0) # OPL
    iroc[:,2] = get_sloped_seam(opl_fil.transpose(), detection_max_slope)

    # incorporate user annotation, else the knowledge that it should be below RPE
    csj_anno = np.equal(user_annos_2d, RetinaLayers.CSJ.value)
    percolreducedanno = csj_anno.any(axis=0)
    csj_anno = np.float64(csj_anno)
    if False: # old approach that didn't use the seam carving
        csj_fil = precomputed_csj * make_mask_for_csj_det(csj_anno, percolreducedanno, irlocs, rpe_width)
        csj_fil -= csj_anno*1e-7 # tiny subtraction fixes case where csj_fil is all zeros and user input would be ignored
        iroc[:,3] = np.argmin(csj_fil, axis=0)
    else: # new approach that uses seam carving
        csj_anno, usronly = make_mask_for_csj_det(csj_anno, percolreducedanno, irlocs, rpe_width)
        csj_fil = (precomputed_csj - 1e-7) * csj_anno - (usronly * 5.)
        iroc[:,3] = get_sloped_seam(csj_fil.transpose(), detection_max_slope)

    if not visualize:
        return iroc

    opl_fil = uint8norm(opl_fil); opl_fil = np.stack((opl_fil,opl_fil,opl_fil), axis=-1)
    csj_fil = uint8norm(csj_fil); csj_fil = np.stack((csj_fil,csj_fil,csj_fil), axis=-1)
    opl_color = uint8clip(np.float32(RetinaLayerColors.OPL.value)[::-1]*255).tolist()
    csj_color = uint8clip(np.float32(RetinaLayerColors.CSJ.value)[::-1]*255).tolist()
    for jj in range(iroc.shape[0]):
        opl_fil[iroc[jj,2],jj] = opl_color
        csj_fil[iroc[jj,3],jj] = csj_color

    return opl_fil, csj_fil, iroc
