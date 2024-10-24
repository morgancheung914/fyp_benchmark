from src.base_model import BaseModel

class MeditronModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config={"Autoregressive": True}, internal_id = "Meditron", model_id = "epfl-llm/meditron-7b")

    def chat_template(self, input):
        return f"{input[0]['content']}\n ###User: {input[1]['content']}\n ###Assistant: "

    def batch_predict(self, input_text, max_length, num_return_seq, temperature):
        inputs = self.tokenizer(self.chat_template(input_text), return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids, max_length=max_length)
        return self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]