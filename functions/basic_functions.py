from ..constants import DIFFUSION_MODELS

def get_model_type(model_id):
    for model in DIFFUSION_MODELS:
        if model.id == model_id:
            return model.type
    return None  # Return None if the ID is not found