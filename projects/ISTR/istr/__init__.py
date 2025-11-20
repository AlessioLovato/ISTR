from .config import add_ISTR_config
from .inseg import ISTR
from .dataset_mapper import ISTRDatasetMapper
from .swin_transformer import build_swint_fpn_backbone
from .dataset_registration import register_dataset_from_args, setup_datasets_from_config