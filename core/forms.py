from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Note, Tag, Resume



class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


class NoteForm(forms.ModelForm):
    """Form for creating and editing notes."""

    # Manual tags input
    manual_tags = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Ajouter des tags (séparés par des virgules)',
            'class': 'input'
        }),
        help_text='Séparez les tags par des virgules'
    )

    class Meta:
        model = Note
        fields = ['title', 'content', 'mood', 'category', 'is_favorite']
        widgets = {
            'title': forms.TextInput(attrs={
                'placeholder': 'Titre de votre note (optionnel)',
                'class': 'input'
            }),
            'content': forms.Textarea(attrs={
                'placeholder': 'Écrivez vos pensées ici...',
                'class': 'input textarea',
                'rows': 10
            }),
            'mood': forms.Select(attrs={
                'class': 'input'
            }),
            'category': forms.Select(attrs={
                'class': 'input'
            }),
            'is_favorite': forms.CheckboxInput(attrs={
                'class': 'checkbox'
            })
        }
        labels = {
            'title': 'Titre',
            'content': 'Contenu',
            'mood': 'Humeur',
            'category': 'Catégorie',
            'is_favorite': 'Marquer comme favori'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add empty option for mood and category
        self.fields['mood'].required = False
        self.fields['category'].required = False

        # Pre-fill manual tags if editing
        if self.instance and self.instance.pk:
            tags = self.instance.tags.all()
            if tags:
                self.initial['manual_tags'] = ', '.join([tag.name for tag in tags])

    def save(self, commit=True):
        note = super().save(commit=False)

        if commit:
            note.save()

            # Handle manual tags
            manual_tags_str = self.cleaned_data.get('manual_tags', '')
            if manual_tags_str:
                tag_names = [name.strip().lower() for name in manual_tags_str.split(',') if name.strip()]

                # Clear existing tags
                note.tags.clear()

                # Add new tags
                for tag_name in tag_names:
                    tag, created = Tag.objects.get_or_create(
                        name=tag_name,
                        defaults={'created_by': note.user, 'is_auto_generated': False}
                    )
                    note.tags.add(tag)
            else:
                note.tags.clear()

        return note


class SearchForm(forms.Form):
    """Form for searching notes."""

    q = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Rechercher dans vos notes...',
            'class': 'input search-input'
        }),
        label='Recherche'
    )

    category = forms.ChoiceField(
        required=False,
        choices=[('', 'Toutes les catégories')] + Note.CATEGORY_CHOICES,
        widget=forms.Select(attrs={'class': 'input'}),
        label='Catégorie'
    )

    mood = forms.ChoiceField(
        required=False,
        choices=[('', 'Toutes les humeurs')] + Note.MOOD_CHOICES,
        widget=forms.Select(attrs={'class': 'input'}),
        label='Humeur'
    )

    tags = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Tags (séparés par des virgules)',
            'class': 'input'
        }),
        label='Tags'
    )



# forms.py

class ResumeGenerateForm(forms.Form):
    PERIOD_CHOICES = [
        ('week', 'Cette semaine'),
        ('month', 'Ce mois'),
        ('year', 'Cette année'),
    ]
    period = forms.ChoiceField(
        choices=PERIOD_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    category = forms.CharField(
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-select', 'placeholder': 'ex: famille, travail...'})
    )