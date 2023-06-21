from transformers import TrainerCallback

class ProfCallback(TrainerCallback):
    """
    推理回调
    """
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()