from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('voice_journal', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='voicejournal',
            name='title',
            field=models.CharField(blank=True, default='', help_text='Custom title for the voice entry', max_length=120),
        ),
    ]


