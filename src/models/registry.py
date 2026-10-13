"""Model registry: name -> Lightning module class, plus the input kind per model."""

import importlib
import inspect

# Models whose input is a 2D image (sequential reshape or spatial/gaussian splat)
IMAGE_MODELS = frozenset({
    'dscnn', 'efficientnet', 'vision_mamba', 'mamba_vision',
    'resnet', 'resnet_cls', 'vit', 'mamba', 'swin', 'spatial_mamba', 'vim_cls',
    'swin_cls', 'dscnn_cls', 'efficientnet_cls', 'vit_cls', 'mamba_cls',
    'mamba_vision_cls', 'spatial_mamba_cls',
})
# Models whose input is a PyG graph
GRAPH_MODELS = frozenset({
    'gcn', 'vgae', 'gcn_cls', 'vgae_cls',
})
# all others take the raw RSSI vector

# name -> (models submodule, class name)
MODEL_REGISTRY = {
    'dscnn':             ('dscnn',           'DSCNN'),
    'efficientnet':      ('efficientnet',    'EfficientNetWiFi'),
    'vision_mamba':      ('vision_mamba',    'VisionMambaWiFi'),
    'mamba_vision':      ('mamba_vision',    'MambaVisionWiFi'),
    'transformer':       ('transformer',     'WiFiTransformerModule'),
    'sae':               ('sae',             'ReferenceSAE'),
    'ffn':               ('ffn',             'ReferenceFFN'),
    'gcn':               ('gcn',             'ReferenceSAGEModule'),
    'vgae':              ('vgae',            'ReferenceGCNVGAEModule'),
    'resnet':            ('resnet',          'ResNetWiFiModule'),
    'vit':               ('vit',             'WiFiViTModule'),
    'mamba':             ('mamba',           'WiFiSpatialMambaModule'),
    'spatial_mamba':     ('spatial_mamba',   'SpatialMambaWiFiModule'),
    'swin':              ('swin',            'SwinWiFiModule'),
    'resnet_cls':        ('cls_variants',    'ResNetClsWiFiModule'),
    'vim_cls':           ('cls_variants',    'VimClsWiFiModule'),
    'swin_cls':          ('cls_variants',    'SwinClsWiFiModule'),
    'dscnn_cls':         ('cls_variants',    'DSCNNClsWiFiModule'),
    'efficientnet_cls':  ('cls_variants',    'EfficientNetClsWiFiModule'),
    'vit_cls':           ('cls_variants',    'ViTClsWiFiModule'),
    'ffn_cls':           ('cls_variants',    'FFNClsWiFiModule'),
    'sae_cls':           ('cls_variants',    'SAEClsWiFiModule'),
    'gcn_cls':           ('cls_variants',    'GCNClsWiFiModule'),
    'vgae_cls':          ('cls_variants',    'VGAEClsWiFiModule'),
    'mamba_cls':         ('cls_variants',    'MambaClsWiFiModule'),
    'mamba_vision_cls':  ('cls_variants',    'MambaVisionClsWiFiModule'),
    'spatial_mamba_cls': ('cls_variants',    'SpatialMambaClsWiFiModule'),
    'transformer_cls':   ('cls_variants',    'TransformerClsWiFiModule'),
}


def get_model_class(model_name: str):
    """Lazy-import and return the Lightning module class for a model name."""
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(
            f"Unknown model: {model_name!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )
    module_name, class_name = entry
    module = importlib.import_module(f'.{module_name}', package=__package__)
    return getattr(module, class_name)


def accepted_params(cls):
    """Return (set of named __init__ params, accepts_var_keyword)."""
    sig = inspect.signature(cls.__init__)
    names = {p.name for p in sig.parameters.values() if p.name != 'self'}
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    return names, has_var_kw
