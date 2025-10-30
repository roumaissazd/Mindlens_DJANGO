from django.contrib import admin
from django.utils.html import format_html
from .models import Note, Tag, Profile  # Ajoutez Profile ici
from .models import Reminder, Resume, PhotoAlbum, Photo

class PhotoInline(admin.TabularInline):
    model = Photo
    extra = 0
    readonly_fields = ('preview',)
    fields = ('preview','person_name','is_user','created_at')
    def preview(self, obj):
        if obj and obj.image:
            from django.utils.html import format_html
            return format_html('<img src="{}" style="height:60px;border-radius:6px;"/>', obj.image.url)
        return ""
    preview.short_description = "Aperçu"

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_auto_generated', 'created_by', 'created_at']
    list_filter = ['is_auto_generated', 'created_at']
    search_fields = ['name']
    ordering = ['name']


@admin.register(Note)
class NoteAdmin(admin.ModelAdmin):
    list_display = ['image_tag', 'get_title', 'user', 'category', 'mood', 'sentiment_label', 'is_favorite', 'created_at']
    list_filter = ['category', 'mood', 'is_favorite', 'sentiment_label', 'created_at']
    search_fields = ['title', 'content', 'user__username']
    readonly_fields = ['image_tag', 'created_at', 'updated_at', 'sentiment_score', 'sentiment_label', 'auto_tags']
    filter_horizontal = ['tags']
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Informations principales', {
            'fields': ('user', 'title', 'content', 'image')  # Ajout de image ici
        }),
        ('Métadonnées', {
            'fields': ('mood', 'category', 'tags', 'is_favorite')
        }),
        ('Analyse IA', {
            'fields': ('sentiment_label', 'sentiment_score', 'auto_tags'),
            'classes': ('collapse',)
        }),
        ('Dates', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def get_title(self, obj):
        return obj.title or f"Note du {obj.created_at.strftime('%d/%m/%Y')}"
    get_title.short_description = 'Titre'

    def image_tag(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-width:70px; max-height:70px; object-fit:cover; border-radius:8px;" />',
                obj.image.url
            )
        return "-"
    image_tag.short_description = 'Image'

    def save_model(self, request, obj, form, change):
        if not change:  # If creating new note
            obj.user = request.user
        super().save_model(request, obj, form, change)


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'photo_user']
    search_fields = ['user__username']

@admin.register(Reminder)
class ReminderAdmin(admin.ModelAdmin):
    # Use actual field names from core.models.Reminder
    list_display = ['user', 'note', 'trigger_at', 'is_read']
    list_filter = ['is_read', 'trigger_at']
    search_fields = ['user__username', 'note__title', 'message']
    readonly_fields = ['created_at']
    date_hierarchy = 'trigger_at'

    fieldsets = (
        ('Informations principales', {
            'fields': ('user', 'note', 'trigger_at', 'is_read', 'priority', 'message')
        }),
        ('Dates', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

@admin.register(PhotoAlbum)
class PhotoAlbumAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'created_at', 'updated_at')
    search_fields = ('name', 'description', 'user__username')
    inlines = [PhotoInline]

@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ('album', 'person_name', 'is_user', 'created_at')
    list_filter = ('is_user','created_at')
    search_fields = ('person_name','album__name')