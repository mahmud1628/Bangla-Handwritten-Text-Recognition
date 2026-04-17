import torch
import torch.nn as nn
from transformers import AutoImageProcessor, GPT2Config, GPT2LMHeadModel, ViTModel

from .config import CHECKPOINT_PATH, VOCAB_FILE
from .tokenizers import BnGraphemizerProcessor


class ViTGPT2EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, max_length=128):
        super().__init__()

        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")

        gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")
        gpt2_config.add_cross_attention = True
        gpt2_config.is_decoder = True

        self.decoder = GPT2LMHeadModel(gpt2_config)
        pretrained_state = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").state_dict()
        self.decoder.load_state_dict(pretrained_state, strict=False)
        self.decoder.resize_token_embeddings(vocab_size)

        vit_hidden_size = self.encoder.config.hidden_size
        gpt2_hidden_size = self.decoder.config.n_embd

        if vit_hidden_size != gpt2_hidden_size:
            self.encoder_projection = nn.Linear(vit_hidden_size, gpt2_hidden_size)
        else:
            self.encoder_projection = nn.Identity()

        self.vocab_size = vocab_size
        self.max_length = max_length

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        max_length=128,
        num_beams=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ):
        encoder_outputs = self.encoder(pixel_values=pixel_values)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = self.encoder_projection(encoder_hidden_states)

        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )

        batch_size = pixel_values.shape[0]
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=pixel_values.device)

        generated = self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            early_stopping=True,
        )

        return generated


def load_recognition_components():
    print("Initializing text recognition components...")

    text_processor = BnGraphemizerProcessor(VOCAB_FILE, model_max_length=128)
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        size={"height": 224, "width": 224},
        use_fast=True,
    )

    device = torch.device("cpu")
    print(f"Using device: {device}")

    recog_model = ViTGPT2EncoderDecoder(vocab_size=len(text_processor.vocab), max_length=128)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    recog_model.load_state_dict(state_dict, strict=False)
    recog_model = recog_model.to(device)
    recog_model.eval()

    print("Recognition model loaded successfully")
    return {
        "text_processor": text_processor,
        "image_processor": image_processor,
        "device": device,
        "recog_model": recog_model,
    }
