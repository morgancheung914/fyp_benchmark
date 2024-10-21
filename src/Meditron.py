from src.base_model import BaseModel

class MeditronModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config={"Autoregressive": True}, internal_id = "Meditron", model_id = "epfl-llm/meditron-7b")

    def predict(self, input_text, max_length, num_return_seq, temperature):
        pass 