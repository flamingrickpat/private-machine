from llama_cpp import Llama

llm = Llama(model_path=r"C:\workspace\AI\gemma\qat\gemma-3-12b-it-q4_0.gguf", n_ctx=1024, lora_path=r"C:\workspace\private-machine\training\adapters\mem_0\Mem_0-F16-LoRA.gguf")
print(llm.create_completion("Q: What color is the sky? A:", max_tokens=128))