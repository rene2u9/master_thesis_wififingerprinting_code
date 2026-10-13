"""Per-sample input transform for a run, selected by input kind (not model name)."""

from typing import List, Optional


def build_transform(
    kind: Optional[str],
    m_cfg: dict,
    data_config: dict,
    db_path: str,
    sorted_aps: List[str],
):
    """kind: 'image' | 'graph' | None (raw RSSI vector)."""
    if kind == 'image':
        image_shape = tuple(m_cfg.get('image_shape', data_config['image_shape']))
        spatial_mode = m_cfg.get('spatial_image_mode', '')
        if spatial_mode == 'gaussian':
            from .image_transforms import RSSIToGaussianSplatImage
            return RSSIToGaussianSplatImage(
                db_path=db_path,
                sorted_aps=sorted_aps,
                image_shape=image_shape,
                sigma=m_cfg.get('gaussian_sigma'),
                sigma_multiplier=m_cfg.get('gaussian_sigma_multiplier', 1.0),
                fill_mode=m_cfg.get('fill_mode', 'zero'),
            )
        if spatial_mode == 'bilinear':
            from .image_transforms import RSSIToBilinearImage
            return RSSIToBilinearImage(
                db_path=db_path,
                sorted_aps=sorted_aps,
                image_shape=image_shape,
                fill_mode=m_cfg.get('fill_mode', 'zero'),
            )
        if spatial_mode == 'upscale':
            from .image_transforms import RSSIToUpscaledImage
            return RSSIToUpscaledImage(
                image_shape=image_shape,
                source_shape=tuple(m_cfg.get('upscale_source_shape', [16, 16])),
                mode=m_cfg.get('upscale_mode', 'nearest'),
            )
        if spatial_mode == 'oa':
            from .image_transforms import RSSIToSpatialImageOA
            return RSSIToSpatialImageOA(
                db_path=db_path,
                sorted_aps=sorted_aps,
                image_shape=image_shape,
            )
        if spatial_mode == 'naive':
            from .image_transforms import RSSIToSpatialImage
            return RSSIToSpatialImage(
                db_path=db_path,
                sorted_aps=sorted_aps,
                image_shape=image_shape,
            )
        from .image_transforms import RSSIToImage
        return RSSIToImage(image_shape=image_shape)

    if kind == 'graph':
        from .graph_transforms import RSSIToGraph
        return RSSIToGraph(
            num_aps=data_config['max_aps'],
            missing_value=data_config['missing_value'],
            rssi_threshold=m_cfg.get('rssi_threshold', -95.0),
            db_path=db_path if m_cfg.get('use_ap_positions', True) else None,
            sorted_aps=sorted_aps,
            graph_topology=m_cfg.get('graph_topology', 'covisible'),
            rssi_normalization=data_config.get('rssi_normalization', 'minmax'),
            rssi_min=data_config.get('rssi_min', -100.0),
            rssi_max=data_config.get('rssi_max', 0.0),
        )

    return None
