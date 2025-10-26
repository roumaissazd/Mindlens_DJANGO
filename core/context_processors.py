from .models import Notification


def notifications(request):
    """
    Context processor to add user notifications to all templates.
    """
    if request.user.is_authenticated:
        # Get unread notifications, limited to 10 most recent
        user_notifications = Notification.objects.filter(
            user=request.user,
            is_read=False
        ).order_by('-timestamp')[:10]

        # Count total unread notifications
        unread_count = Notification.objects.filter(
            user=request.user,
            is_read=False
        ).count()

        return {
            'notifications': user_notifications,
            'unread_notifications_count': unread_count,
        }
    return {
        'notifications': [],
        'unread_notifications_count': 0,
    }