import torch
from transformers import (
    AutoTokenizer,
)

from bottleneck_t5 import BottleneckT5LMWithPerturbV2


class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.model = BottleneckT5LMWithPerturbV2.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        return self.model(
            **inputs,
            decoder_input_ids=decoder_inputs['input_ids'],
            encode_only=True,
        )[0]

    @torch.no_grad()
    def generate_from_latent(self, latent: torch.FloatTensor, max_length=512, temperature=1.0) -> str:
        dummy_text = '.'
        dummy = self.embed(dummy_text)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector
        input_ids = self.tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
            use_cache=False
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
