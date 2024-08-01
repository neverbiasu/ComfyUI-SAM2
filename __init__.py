from .sam2 import *

NODE_CLASS_MAPPINGS = {
    'SAM2ModelLoader (segment anything)': SAM2ModelLoader,
    'GroundingDinoModelLoader (segment anything)': GroundingDinoModelLoader,
    'GroundingDinoSAMSegment (segment anything)': GroundingDinoSAMSegment,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'SAM2ModelLoader': 'SAM2 Model Loader',
    'GroundingDinoModelLoader': 'Grounding Dino Model Loader',
    'GroundingDinoSAMSegment': 'Grounding Dino SAM Segment',
    'InvertMask': 'Invert Mask',
    "IsMaskEmpty": "Is Mask Empty",
}
