from src.base_model import BaseModel
import torch
class InternistModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config={"Autoregressive": True}, internal_id = "Internist", model_id = "internistai/base-7b-v0.2")

    def predict(self, input_text, max_length, num_return_seq, temperature, top_p):


        
        encodeds = self.tokenizer.apply_chat_template(input_text, add_generation_prompt=True ,return_tensors="pt")

        model_inputs = encodeds.to(self.device)
        

        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded

    def batch_predict(self, input_text, max_length, num_return_seq, temperature, top_p):
        
        """
        Text generation in batch.

        Args:
            input_text List[str]: the batch of input text in list
        """


        encoded = self.tokenizer.apply_chat_template(input_text, add_generation_prompt=True, padding = True, return_tensors="pt")

        
        
        model_inputs = encoded.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=max_length, do_sample=True)

       
        generated_ids = generated_ids[:,encoded.shape[-1]:]
        decoded = self.tokenizer.batch_decode(generated_ids)
        
        return decoded