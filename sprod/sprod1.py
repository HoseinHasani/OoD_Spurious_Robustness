from .base_sprod import BaseSPROD

class SPROD1(BaseSPROD):
    def __init__(self, config):
        super().__init__(config, probabilistic_score=False, normalize_features=True)
