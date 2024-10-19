from src.base_model import BaseModel
class Med42Model(BaseModel):
    
    def __init__(self, config=None):
        super().__init__(config={"Autoregressive": True}, internal_id = "Med42", model_id = "m42-health/Llama3-Med42-8B")
        
    def predict(self, input_text, max_length, num_return_seq, temperature):

        input_ids = self.tokenizer.apply_chat_template(
        input_text,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(self.device)

        terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
        outputs = self.model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9)
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)

        return decoded_response
    
    def batch_predict(self, input_text, max_length, num_return_seq, temperature):

        
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        input_ids = self.tokenizer.apply_chat_template(
        input_text,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt"
    ).to(self.device)

        terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
        outputs_all = self.model.generate(   
    input_ids,
    max_new_tokens=max_length,
    eos_token_id=terminators,
    do_sample=True,
    temperature=temperature,
    top_p=0.9)

        
        
        response = outputs_all[:,input_ids.shape[-1]:]
        
        decoded_response = self.tokenizer.batch_decode(response, skip_special_tokens=True)

        return decoded_response