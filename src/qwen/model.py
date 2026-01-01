import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLocal:
    """
    Local Qwen2.5 runner for Apple Silicon (MPS).
    Uses a project-local Hugging Face cache and exposes full
    generation parameter control using FP16 weights.
    """

    def __init__(
        self,
        modelId: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "mps",
        cacheDir: str = "huggingFaceCache",
    ):
        # Force Hugging Face to use project-local cache
        os.environ["HF_HOME"] = os.path.abspath(cacheDir)

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this system.")

        self.device = device

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelId,
            trust_remote_code=True,
        )

        # Model (FP16 for Apple Silicon)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelId,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Move model to Metal
        self.model.to(self.device)
        self.model.eval()

    def chat(
        self,
        messages: list[dict],
        *,
        maxNewTokens: int = 300,
        temperature: float = 0.2,
        topP: float = 0.9,
        topK: int = 50,
        repetitionPenalty: float = 1.1,
        doSample: bool | None = None,
    ) -> str:
        """
        Generate a response from chat-style messages.

        messages format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """

        if doSample is None:
            doSample = temperature > 0.0

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=maxNewTokens,
                temperature=temperature,
                top_p=topP,
                top_k=topK,
                repetition_penalty=repetitionPenalty,
                do_sample=doSample,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)