"""Hits to waveform like data converter
"""

import numpy as np

from .data_containers import (
    Current,
    EffectiveCharge,
    EventData,
    Geometry,
    Hits,
    ReadoutConfig,
)


def hits_to_bin_wf(hits: Hits, template: np.ndarray, threshold: float, bin_size: int, nburst: int, npad=10) -> None:
    """Convert hit data to waveform-like data structure.

    Timestamps of hits and template are sampled at the same accuracy.

    Hits are assumed to be always positive.

    Template is fine sampled. Hits is coarsely sampled.

    Template is assumed to positive.

    Convert the hit data into a waveform-like format using the provided template.

    Waveform-like format is on coarse grained time bins but fine grained time accuracy.

    Integration of charge is done per coarse grained time bins.

    Returns:
        None
    """
    n = len(hits)
    uq_pxl = np.unique(hits.location[:, :2], axis=0)
    if len(uq_pxl) != n:
        raise ValueError("Duplicate pixel locations found in hits and not supported.")
    recorded_charges = np.zeros((n, nburst)) #
    recorded_charges[:, :] = hits.data[:, 3:]  # Assuming hits.data columns are [x, y, z, q1, q2, ..., qnburst], with noise

    fracs = threshold/np.max(recorded_charges[:, :], axis=1)
    print('fracs', fracs)

    cum_template = np.cumsum(template)
    fracs = np.where(fracs<1, fracs, 1.0)
    # first closet point in cum_template
    # assume cum_template is monotonically increasing
    if (template<-1E-6).any():
        raise ValueError("Template contains negative values, which is not allowed.")
    indices = np.searchsorted(cum_template, fracs)

    ind_max = np.max(indices) + 1
    # round to integer multiple of bin_size
    ind_max = int(np.ceil(ind_max/bin_size)*bin_size)
    print(ind_max, indices, len(template))

    j = np.arange(ind_max)                      # 0..ind_max-1
    k = indices[:, None]                     # (n, 1)
    # positions in cum_template to read from for each output column
    src = j - (ind_max - k)                     # (n, ind_max)

    trunc = np.zeros((n, ind_max), dtype=cum_template.dtype)
    mask = (src >= 0) & (src <= k)            # valid copied region
    trunc[mask] = cum_template[src[mask]]

    # scale out by threshold

    trunc *= threshold/trunc[:, -1][:, None]

    trunc_coarse = trunc[:, bin_size-1::bin_size]
    nt_trunc = trunc_coarse.shape[-1]

    wf_out_data = np.zeros((n, nburst+nt_trunc), dtype=trunc_coarse.dtype)
    wf_out_data[:, nt_trunc:] = recorded_charges
    wf_out_data[:, :nt_trunc] = trunc_coarse
    wf_out_data = np.diff(wf_out_data, axis=1, prepend=0)
    wf_out_data = np.pad(wf_out_data, ((0, 0), (npad, npad)), mode='constant', constant_values=0)

    wf_out_location = np.zeros((n, 3), dtype=hits.location.dtype)
    wf_out_location[:, :2] = hits.location[:, :2]
    wf_out_location[:, 2] = hits.location[:, 2] - ind_max - npad * bin_size

    # To return charge per integral
    wf = Current(data=wf_out_data, location=wf_out_location,
                 tpc_id=hits.tpc_id, event_id=hits.event_id)
    return wf


def convert_bin_wf_to_blocks(wf: Current, bin_size: int,
                             shift_to_center: bool = False) -> Current:
    """Convert binned waveform data to block format.

    The original waveform data is assumed to be in coarse grained time bins.
    The original waveform location is assumed to be on fine grained time accuracy.

    The output block data is in fine grained time bins after rounding.
    """

    # check no duplicates.
    wfloc = wf.location
    print('------------------------ loc', wfloc)
    # shift to coarse grained bin center
    if shift_to_center:
        wfloc[:, 2] = (wfloc[:, 2] + 0.5*bin_size) // bin_size
    else:
        wfloc[:, 2] = wfloc[:, 2] // bin_size
    uq_wfloc, inv_wfloc, nc = np.unique(wfloc, axis=0, return_inverse=True,
                                        return_counts=True)
    if np.any(nc > 1):
        raise ValueError("Duplicate waveform locations found, cannot convert to blocks.")

    # FIXME: a global block is returned.
    # Find the shape
    loc_min = np.min(wfloc, axis=0)
    loc_max = np.max(wfloc, axis=0)
    shape = loc_max - loc_min + 1
    # index mapping
    local_ind = wfloc - loc_min
    nt = wf.data.shape[-1]
    local_ind_full = np.broadcast_to(local_ind[:, None, :], (local_ind.shape[0], nt, 3)).copy()
    local_ind_full[:, :, -1] += np.arange(nt)[None, :]

    # location
    bloc = loc_min[None, :]
    bloc[:, -1] = bloc[:, -1]* bin_size
    # allocate bdata
    bdata = np.zeros((*shape[:2], nt), dtype=wf.data.dtype)
    # extend local_ind to length of time axis by + np.arange
    # np.add.at with local_ind_full, wf.data
    indices = (local_ind_full[:, :, 0].flatten(),
                      local_ind_full[:, :, 1].flatten(),
                      local_ind_full[:, :, 2].flatten())
    bata = np.add.at(bdata, indices, wf.data.flatten())

    block = Current(data=bdata[None, ...], location=bloc,
                    tpc_id=wf.tpc_id, event_id=wf.event_id)
    return block
