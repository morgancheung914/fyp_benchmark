from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 


class BaseModel:
    def __init__(self, config, internal_id, model_id=None):
        """
        Args:
            config (Dict): base configuration during model initialization.
            model_id (str): model_id on huggingface

        """
        self.config = config 
        self.model = None 
        self.tokenizer = None
        self.model_id = model_id
        self.internal_id = internal_id
        print(f">Bench> Model {internal_id} initiated.")

    def load_model(self):
        """
        Load the model from the given hf_id.
        
        """
        cache_dir = "./models"
        # TODO: allow config device from config files
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        
        if ("Autoregressive" in self.config) and (self.config["Autoregressive"] == True):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir = cache_dir, padding_side = 'left')
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir = cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir = cache_dir, device_map="auto")
        
        print(f">Bench> Model {self.internal_id} loaded.")

        #self.model.to(self.device)

        print(f">Bench> Model {self.internal_id} loaded onto {self.device}.")
    
    def predict(self, input_text, max_length, num_return_seq, temperature):
        

        """
        Generation for singular input_text.

        Args: 
            input_text (str): the input text IN CHAT FORMAT
        """
        
        #input_ids = self.tokenizer.encode(input_text, return_tensors = 'pt').to(self.device)
        input_ids = self.tokenizer.encode(input_text, return_tensors = 'pt')

        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_seq, temperature=temperature)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text
        
    
    #batch predict