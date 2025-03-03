import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import time
class LLM_Model:
    def __init__(self, model_name,token=None):
        self.token = token
        self.model_name = model_name
        self.device = None
        self.device_map = None
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.system_prompt= "you are helpful assistant."
        self.setup_device()


    def setup_device(self):
        """Set up the device (GPU/CPU) for model usage."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_map = {"": 0}  # Load the model onto GPU
        else:
            self.device = torch.device("cpu")
            self.device_map = {"": "cpu"}  # Load the model onto CPU

    def setup_model_and_tokenizer(self):
        """Load the model and tokenizer with the appropriate configuration."""
        
        use_4bit = torch.cuda.is_available()
        bnb_4bit_compute_dtype = "float16" if torch.cuda.is_available() else "float32"
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        # Setup BitsAndBytes configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map=self.device_map,
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer
        

    def generate_text(self, prompt):
        """Generate text using the loaded model and tokenizer."""

        messages = [
            {"role": "system", "content":self.system_prompt },
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        print("Response Generating...")
        start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=200,
            temperature=0.2,
            

        )
        end_time = time.time()-start_time
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        gen_token_per_sec = len(generated_ids[0]) /end_time
        print(50*"*")
        print("Total token :",len(generated_ids[0]))
        print("Total time Taken:",end_time)
        print("generated token per second:",gen_token_per_sec)
        print(50*"*")
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

if __name__ == "__main__":
    system_prompt="You are a helpful assistant."
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "what is atom?"
    handler = LLM_Model( model_name=model_name)
    handler.setup_model_and_tokenizer()
    result = handler.generate_text(prompt)
    print(result)
