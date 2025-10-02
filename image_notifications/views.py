from django.shortcuts import render

# Create your views here.
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from core.models import Note  # Adjust import
from .utils import generate_image_from_note
from .models import GeneratedImage

@login_required
def generate_image_for_note(request, note_id):
    note = get_object_or_404(Note, id=note_id, user=request.user)
    if not note.generated_image_obj:
        image_file = generate_image_from_note(note)
        if image_file:
            GeneratedImage.objects.update_or_create(
                note=note,
                defaults={'image_file': image_file}
            )
    return redirect('note_detail', note_id=note.id)  # Adjust URL name