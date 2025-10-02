import torch
from diffusers import StableDiffusionPipeline
from django.conf import settings
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile
import logging

logger = logging.getLogger(__name__)

def generate_image_from_note(note):
    try:
        # Load Stable Diffusion model
        pipe = StableDiffusionPipeline.from_pretrained(
            getattr(settings, 'STABLE_DIFFUSION_MODEL', 'CompVis/stable-diffusion-v1-4')
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Craft prompt from note data
        prompt = f"A {note.mood} scene representing {note.title or note.content[:50]} in {note.category} context"
        if note.sentiment_label:
            prompt += f", with a {note.sentiment_label} mood"

        # Generate image
        image = pipe(prompt).images[0]
        img_io = BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)

        return ContentFile(img_io.read(), name=f"generated_{note.id}.png")
    except Exception as e:
        logger.error(f"Error generating image for note {note.id}: {e}")
        return None