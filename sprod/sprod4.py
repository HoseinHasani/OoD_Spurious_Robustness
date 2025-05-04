from full_sprod import FullSPROD

class SPROD4(FullSPROD):
    def __init__(self, config):
        super().__init__(config,
                         use_group_refinement=True,
                         merge_refined_prototypes=True,
                         refine_n_iter=1)
