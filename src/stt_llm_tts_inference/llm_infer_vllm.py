import torch
from vllm import LLM, SamplingParams

class DeepSeekInference:
    def __init__(self, model_path: str):
        """
        Initialize the DeepSeekInference class with the model path.
        
        Args:
            model_path (str): Path to the locally stored model.
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Load the model from the specified path using vllm.
        """
        # Load the model using vllm
        self.model = LLM(model=self.model_path,device="auto",max_model_len=200, enforce_eager=True)
        print(f"Model loaded from {self.model_path}")
    
    def infer(self, prompt: str):
        """
        Perform inference using the loaded model.
        
        Args:
            prompt (str): The input text prompt to generate predictions.
        
        Returns:
            str: The generated text.
        """
        if not self.model:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Define sampling parameters (e.g., temperature, top_k, etc.)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, num_beams=1, max_tokens=200)
        
        # Run inference on the model
        output = self.model.generate(prompt, sampling_params=sampling_params)
        
        # Return the generated text
        return output

if __name__ == "__main__":
    # Path to your locally stored model (update this with your actual model path)
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Initialize and load the model
    inference = DeepSeekInference(model_path=model_path)
    inference.load_model()

    # Provide a prompt to generate text
    prompt = "What is the future of AI in medicine?"
    
    # Get the model's prediction
    generated_text = inference.infer(prompt)

    # Print the generated text
    print(f"Generated text: {generated_text}")
