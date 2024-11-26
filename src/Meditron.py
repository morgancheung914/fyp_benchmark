from src.base_model import BaseModel

class MeditronModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config={"Autoregressive": True}, internal_id = "Meditron", model_id = "epfl-llm/meditron-7b")

    def chat_template(self, input_text):
        # Meditron (fine-tuned on Llama2 requires a specific format)
        sys_prompt = "You are a helpful medical assistant. "
        sys_prompt += f"{input_text[0]['content']}\n"

        sys_prompt += f"### User: {input_text[1]['content']}\n### Assistant:"
        #print(sys_prompt)
        return sys_prompt

    def batch_predict(self, input_text, max_length, num_return_seq, temperature, top_p):
        res = []
        for i in input_text:
            processed_prompt = self.chat_template(i)
   
            inputs = self.tokenizer(processed_prompt, return_tensors="pt")["input_ids"]

            generate_ids = self.model.generate(inputs, max_new_tokens=max_length)
            response = self.tokenizer.decode(generate_ids[0][inputs.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #print(f"Model response: {response}")
            res.append(response)
        return res