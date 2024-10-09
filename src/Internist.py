from src.base_model import BaseModel

class InternistModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config, internal_id = "Internist", model_id = "internistai/base-7b-v0.2")

    def predict(self, input_text, max_length, num_return_seq, temperature):

        messages = [
    {"role": "user", "content": "Describe the anatomy of nutcracker syndrome"},
    ]

        encodeds = self.tokenizer.apply_chat_template(input_text, add_generation_prompt=True ,return_tensors="pt")

        model_inputs = encodeds.to(self.device)
        

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded
