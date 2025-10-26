/**
 * Notes Module JavaScript
 * Handles interactive features: favorites, auto-save, animations
 */

// ============ FAVORITE TOGGLE ============
document.addEventListener('DOMContentLoaded', function() {
  // Handle favorite buttons
  const favoriteButtons = document.querySelectorAll('.btn-favorite, .btn-favorite-detail');

  favoriteButtons.forEach(button => {
    button.addEventListener('click', async function(e) {
      e.preventDefault();
      e.stopPropagation();

      const noteId = this.dataset.noteId;

      try {
        const response = await fetch(`/notes/${noteId}/toggle-favorite/`, {
          method: 'POST',
          headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json'
          }
        });

        if (response.ok) {
          const data = await response.json();

          // Toggle active class
          this.classList.toggle('active');

          // Update title
          if (data.is_favorite) {
            this.title = 'Retirer des favoris';
          } else {
            this.title = 'Ajouter aux favoris';
          }

          // Show feedback
          showNotification(
            data.is_favorite ? '⭐ Ajouté aux favoris' : 'Retiré des favoris',
            'success'
          );
        }
      } catch (error) {
        console.error('Error toggling favorite:', error);
        showNotification('❌ Erreur lors de la mise à jour', 'error');
      }
    });
  });

  // Handle completed buttons
  const completedButtons = document.querySelectorAll('.btn-completed');

  completedButtons.forEach(button => {
    button.addEventListener('click', async function(e) {
      e.preventDefault();
      e.stopPropagation();

      const noteId = this.dataset.noteId;

      try {
        const response = await fetch(`/notes/${noteId}/toggle-completed/`, {
          method: 'POST',
          headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json'
          }
        });

        if (response.ok) {
          const data = await response.json();

          // Toggle active class
          this.classList.toggle('active');

          // Update icon and title
          if (data.is_completed) {
            this.innerHTML = '✅';
            this.title = 'Marquer comme non terminé';
          } else {
            this.innerHTML = '⏳';
            this.title = 'Marquer comme terminé';
          }

          // Show feedback
          showNotification(
            data.is_completed ? '✅ Note marquée comme terminée' : '⏳ Note marquée comme en cours',
            'success'
          );
        }
      } catch (error) {
        console.error('Error toggling completed:', error);
        showNotification('❌ Erreur lors de la mise à jour', 'error');
      }
    });
  });
});

// ============ AUTO-SAVE FUNCTIONALITY ============
let autoSaveTimeout;
const AUTO_SAVE_DELAY = 30000; // 30 seconds

function initAutoSave() {
  const form = document.getElementById('noteForm');
  const contentField = document.querySelector('textarea[name="content"]');
  const indicator = document.getElementById('autosaveIndicator');
  
  if (!form || !contentField) return;
  
  // Save to localStorage
  contentField.addEventListener('input', function() {
    clearTimeout(autoSaveTimeout);
    
    autoSaveTimeout = setTimeout(() => {
      const formData = {
        title: document.querySelector('input[name="title"]')?.value || '',
        content: contentField.value,
        mood: document.querySelector('select[name="mood"]')?.value || '',
        category: document.querySelector('select[name="category"]')?.value || '',
        manual_tags: document.querySelector('input[name="manual_tags"]')?.value || ''
      };
      
      localStorage.setItem('note_draft', JSON.stringify(formData));
      
      // Update indicator
      if (indicator) {
        indicator.style.background = '#d1fae5';
        indicator.querySelector('.autosave-text').textContent = 'Brouillon sauvegardé';
        
        setTimeout(() => {
          indicator.querySelector('.autosave-text').textContent = 'Sauvegarde automatique activée';
        }, 2000);
      }
    }, AUTO_SAVE_DELAY);
  });
  
  // Restore from localStorage on page load
  const savedDraft = localStorage.getItem('note_draft');
  if (savedDraft && !form.dataset.editing) {
    try {
      const draft = JSON.parse(savedDraft);
      
      if (confirm('Un brouillon a été trouvé. Voulez-vous le restaurer ?')) {
        if (draft.title) document.querySelector('input[name="title"]').value = draft.title;
        if (draft.content) contentField.value = draft.content;
        if (draft.mood) document.querySelector('select[name="mood"]').value = draft.mood;
        if (draft.category) document.querySelector('select[name="category"]').value = draft.category;
        if (draft.manual_tags) document.querySelector('input[name="manual_tags"]').value = draft.manual_tags;
        
        // Trigger character count update
        const event = new Event('input');
        contentField.dispatchEvent(event);
      }
    } catch (error) {
      console.error('Error restoring draft:', error);
    }
  }
  
  // Clear draft on successful submit
  form.addEventListener('submit', function() {
    localStorage.removeItem('note_draft');
  });
}

// Initialize auto-save if on form page
if (document.getElementById('noteForm')) {
  initAutoSave();
}

// ============ ANIMATIONS ============
// Fade in cards on scroll
const observerOptions = {
  threshold: 0.1,
  rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '0';
      entry.target.style.transform = 'translateY(20px)';
      
      setTimeout(() => {
        entry.target.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }, 100);
      
      observer.unobserve(entry.target);
    }
  });
}, observerOptions);

// Observe note cards
document.querySelectorAll('.note-card, .stat-card, .chart-card').forEach(card => {
  observer.observe(card);
});

// ============ UTILITY FUNCTIONS ============
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

function showNotification(message, type = 'info') {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  
  // Style
  Object.assign(notification.style, {
    position: 'fixed',
    top: '20px',
    right: '20px',
    padding: '16px 24px',
    borderRadius: '12px',
    background: type === 'success' ? '#d1fae5' : '#fee2e2',
    color: type === 'success' ? '#065f46' : '#991b1b',
    fontWeight: '600',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
    zIndex: '9999',
    animation: 'slideIn 0.3s ease',
    maxWidth: '300px'
  });
  
  // Add to page
  document.body.appendChild(notification);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
      notification.remove();
    }, 300);
  }, 3000);
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from {
      transform: translateX(400px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  @keyframes slideOut {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(400px);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);

// ============ SEARCH ENHANCEMENTS ============
// Add debounce to search input
const searchInput = document.querySelector('.search-input');
if (searchInput) {
  let searchTimeout;
  
  searchInput.addEventListener('input', function() {
    clearTimeout(searchTimeout);
    
    // Show loading indicator
    this.style.borderColor = '#F5A623';
    
    searchTimeout = setTimeout(() => {
      this.style.borderColor = '#e5e7eb';
    }, 500);
  });
}

// ============ FORM VALIDATION ============
const noteForm = document.getElementById('noteForm');
if (noteForm) {
  noteForm.addEventListener('submit', function(e) {
    const content = document.querySelector('textarea[name="content"]').value.trim();
    
    if (content.length < 10) {
      e.preventDefault();
      showNotification('⚠️ Le contenu doit contenir au moins 10 caractères', 'error');
      return false;
    }
    
    // Show loading state
    const submitBtn = this.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyse en cours...';
    }
  });
}

// ============ KEYBOARD SHORTCUTS ============
document.addEventListener('keydown', function(e) {
  // Ctrl/Cmd + S to save (on form page)
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    const form = document.getElementById('noteForm');
    if (form) {
      e.preventDefault();
      form.submit();
    }
  }
  
  // Ctrl/Cmd + K to focus search
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
      e.preventDefault();
      searchInput.focus();
    }
  }
});

// ============ SMOOTH SCROLL ============
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const href = this.getAttribute('href');
    if (href !== '#' && href !== '#!') {
      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    }
  });
});

// ============ NOTIFICATIONS DROPDOWN ============
document.addEventListener('DOMContentLoaded', function() {
  const notificationToggle = document.getElementById('notification-toggle');
  const notificationsMenu = document.getElementById('notifications-menu');
  const markAllReadLink = document.querySelector('.mark-all-read');

  if (notificationToggle && notificationsMenu) {
    // Toggle dropdown
    notificationToggle.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      notificationsMenu.classList.toggle('show');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
      if (!notificationToggle.contains(e.target) && !notificationsMenu.contains(e.target)) {
        notificationsMenu.classList.remove('show');
      }
    });

    // Mark all as read
    if (markAllReadLink) {
      markAllReadLink.addEventListener('click', async function(e) {
        e.preventDefault();

        try {
          const response = await fetch('/api/notifications/mark-all-read/', {
            method: 'POST',
            headers: {
              'X-CSRFToken': getCookie('csrftoken'),
              'Content-Type': 'application/json'
            }
          });

          if (response.ok) {
            // Hide notification badge
            const badge = document.querySelector('.notification-badge');
            if (badge) badge.style.display = 'none';

            // Mark all items as read visually
            document.querySelectorAll('.notification-item.unread').forEach(item => {
              item.classList.remove('unread');
              // Remove mark as read button
              const markBtn = item.querySelector('.mark-read-btn');
              if (markBtn) markBtn.remove();
            });

            // Update unread count
            const unreadCount = document.querySelector('.unread-count');
            if (unreadCount) unreadCount.textContent = '0 non lues';

            showNotification('✅ Toutes les notifications marquées comme lues', 'success');
          }
        } catch (error) {
          console.error('Error marking notifications as read:', error);
          showNotification('❌ Erreur lors de la mise à jour', 'error');
        }
      });
    }

    // Handle clicking on notification items (they are now links)
    // The click will navigate to the mark as read URL, so no additional JS needed
    // The page will redirect and mark as read automatically
  }
});

console.log('✨ MindLense Notes Module loaded successfully');

