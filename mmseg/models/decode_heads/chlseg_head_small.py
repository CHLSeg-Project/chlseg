# CHLSegHead_small: identical logic to CHLSegHead_tiny -- all module widths
# are derived from self.in_channels at runtime, so the same head works for
# MSCAN-S's [64, 128, 320, 512] without code changes. This file exists so
# small-variant configs can name `type='CHLSegHead_small'` for clarity in
# logs and work_dirs, and so future small-only tweaks have a place to live
# without forking the tiny class.
from ..builder import MODELS
from .chlseg_head import CHLSegHead_tiny


@MODELS.register_module()
class CHLSegHead_small(CHLSegHead_tiny):
    pass
