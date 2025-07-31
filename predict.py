import torch
from cog import BasePredictor, Input, Path
from transformers import AutoProcessor, VoxtralForConditionalGeneration

DEVICE = "cuda"
REPO_ID = "mistralai/Voxtral-Small-24B-2507"


class Predictor(BasePredictor):
    def setup(self):
        self.processor = AutoProcessor.from_pretrained(
            REPO_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        self.model = VoxtralForConditionalGeneration.from_pretrained(REPO_ID)
        self.model.to(DEVICE)

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        language: str = Input(
            description="Language code for transcription",
            choices=["en", "es", "fr", "de", "it", "nl", "pt", "hi"],
            default="en",
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=1024,
            ge=1,
            le=32768,
        ),
    ) -> str:
        try:
            inputs = self.processor.apply_transcrition_request(
                language=language, audio=str(audio), model_id=REPO_ID
            )
            inputs = inputs.to(DEVICE, dtype=torch.bfloat16)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            decoded_outputs = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            return decoded_outputs[0].strip()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
