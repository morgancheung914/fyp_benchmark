from src.base_model import BaseModel
class PanaceaModel(BaseModel):
    
    def __init__(self, config=None):
        super().__init__(config={"Autoregressive": True}, internal_id = "Panacea", model_id = "wzqacky/Llama-3.1-Panacea-8B-Instruct")
        
    def predict(self, input_text, max_length, num_return_seq, temperature, top_p):
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
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

    
    def batch_predict(self, input_text, max_length, num_return_seq, temperature, top_p):
        formatted_inputs = self.tokenizer.apply_chat_template(
        input_text,
        tokenize=False,
        add_generation_prompt=True,
    )

        inputs = self.tokenizer(formatted_inputs, return_tensors='pt', padding=True).to(self.device)

        outputs_all = self.model.generate(   
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=max_length,
    do_sample=True,
    temperature=temperature,
    eos_token_id=self.tokenizer.eos_token_id, # set eos_token, so that the model knows when the stop generating
    num_return_sequences = num_return_seq,
    top_p=0.9)
    
        response = outputs_all[:,inputs['input_ids'].shape[-1]:]
        decoded_response = self.tokenizer.batch_decode(response, skip_special_tokens=True)
        return decoded_response