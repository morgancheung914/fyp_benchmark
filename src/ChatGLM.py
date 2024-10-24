from src.base_model import BaseModel
class ChatGLMModel(BaseModel):
    
    def __init__(self, config=None):
        super().__init__(config={}, internal_id = "ChatGLM", model_id = "THUDM/chatglm3-6b") 
        
    def predict(self, input_text, max_length, num_return_seq, temperature):

        input_ids = self.tokenizer.build_chat_template(
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
        responses = []
        for text in input_text:


            res, history = self.model.chat(self.tokenizer, "", history=text)
            
            
            responses.append(res)
        #     terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        #     outputs = self.model.generate(
        # input_ids,
        # max_new_tokens=256,
        # eos_token_id=terminators,
        # do_sample=True,
        # temperature=0.6,
        # top_p=0.9)
        #     response = outputs[0][input_ids.shape[-1]:]
        #     decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
        #     responses.append(decoded_response)
        return responses

    
    