from src.tti.dpar import AbsoluteRealityTextToImage
from src.tti.mock import MockTextToImage
from src.tti.sd2_realistic import SdTwoRealisticTextToImage
from src.tti.sdxl import SdxlTextToImage
from src.tti.flux import FluxTextToImage
from src.tti.sd3 import Sd3TextToImage

name_to_tti_mapping = {
    "dpar": AbsoluteRealityTextToImage,
    "sdxl": SdxlTextToImage,
    "mock": MockTextToImage,
    "sd2r": SdTwoRealisticTextToImage,
    "flux": FluxTextToImage,
    "mock": MockTextToImage,
    "sd3": Sd3TextToImage
}