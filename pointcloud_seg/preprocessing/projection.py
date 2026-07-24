import numpy as np
import laspy


def write_labelled_las(
    full_xyz: np.ndarray,
    full_rgb: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    source_header=None,
):
    """
    Write a LAS file with the classification field set to labels.

    When `source_header` (a laspy LasHeader) is given, the output preserves the source
    georeferencing: offsets, scales, and the CRS VLRs are copied so the cloud lands at the
    correct projected coordinates (and large coordinates can't overflow int32). Without it,
    offsets are derived from the data to stay overflow-safe, but no CRS is attached.
    """
    header = laspy.LasHeader(point_format=7, version="1.4")
    if source_header is not None:
        header.offsets = source_header.offsets
        header.scales = source_header.scales
        # pyproj-free CRS preservation: copy the WKT / GeoTIFF-key CRS VLRs directly.
        CRS_IDS = (2111, 2112, 34735, 34736, 34737)
        copied_wkt = False
        for vlr in source_header.vlrs:
            if getattr(vlr, "record_id", None) in CRS_IDS:
                header.vlrs.append(vlr)
                copied_wkt |= vlr.record_id in (2111, 2112)
        if copied_wkt:
            header.global_encoding.wkt = 1   # required for point formats >= 6
    else:
        # No source header: derive offsets from the data so large (projected) coordinates
        # stay within the int32 storage range.
        header.offsets = np.floor(full_xyz.min(axis=0))
        header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header=header)

    las.x = full_xyz[:, 0]
    las.y = full_xyz[:, 1]
    las.z = full_xyz[:, 2]

    # RGB in LAS is 16-bit
    rgb_16 = (np.clip(full_rgb, 0.0, 1.0) * 65535).astype(np.uint16)
    las.red = rgb_16[:, 0]
    las.green = rgb_16[:, 1]
    las.blue = rgb_16[:, 2]

    las.classification = labels.astype(np.uint8)

    las.write(output_path)
