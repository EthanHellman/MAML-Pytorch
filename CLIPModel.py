import torch
import open_clip

class CLIPModel:
    def __init__(self, model_name='ViT-L-14', checkpoint_path=None):
        # Load the CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)

        if checkpoint_path:
            # Load the checkpoint if provided
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(ckpt)

        self.model = self.model.cuda().eval()

    def encode_image(self, image):
        # Encode an image using the CLIP model
        return self.model.encode_image(image)

    def encode_text(self, text):
        # Encode text using the CLIP model
        return self.model.encode_text(text)

    def preprocess_image(self, image):
        # Preprocess an image using the CLIP model's preprocessing
        return self.preprocess(image)
