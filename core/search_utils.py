"""
Search utilities using Whoosh for full-text search on notes.
"""

import os
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME, KEYWORD
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import And, Or, Term
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


def get_schema():
    """Define the search schema for notes."""
    return Schema(
        note_id=ID(stored=True, unique=True),
        user_id=ID(stored=True),
        title=TEXT(stored=True),
        content=TEXT(stored=True),
        category=KEYWORD(stored=True, commas=True),
        mood=KEYWORD(stored=True, commas=True),
        tags=KEYWORD(stored=True, commas=True),
        created_at=DATETIME(stored=True),
    )


def get_index():
    """Get or create the Whoosh index."""
    index_dir = settings.WHOOSH_INDEX_DIR
    
    # Create directory if it doesn't exist
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    # Create or open index
    if index.exists_in(index_dir):
        return index.open_dir(index_dir)
    else:
        return index.create_in(index_dir, get_schema())


def index_note(note):
    """
    Index a single note.
    
    Args:
        note: Note model instance
    """
    try:
        ix = get_index()
        writer = ix.writer()
        
        # Get tags as comma-separated string
        tags_str = ','.join([tag.name for tag in note.tags.all()])
        
        # Add auto-generated tags
        auto_tags = note.get_auto_tags_list()
        if auto_tags:
            tags_str += ',' + ','.join(auto_tags)
        
        writer.update_document(
            note_id=str(note.id),
            user_id=str(note.user.id),
            title=note.title or '',
            content=note.content,
            category=note.category or '',
            mood=note.mood or '',
            tags=tags_str,
            created_at=note.created_at,
        )
        
        writer.commit()
        logger.info(f"Indexed note {note.id}")
    except Exception as e:
        logger.error(f"Error indexing note {note.id}: {e}")


def remove_note_from_index(note_id):
    """
    Remove a note from the index.
    
    Args:
        note_id: ID of the note to remove
    """
    try:
        ix = get_index()
        writer = ix.writer()
        writer.delete_by_term('note_id', str(note_id))
        writer.commit()
        logger.info(f"Removed note {note_id} from index")
    except Exception as e:
        logger.error(f"Error removing note {note_id} from index: {e}")


def search_notes(user_id, query_text=None, category=None, mood=None, tags=None, limit=50):
    """
    Search notes with various filters.
    
    Args:
        user_id: ID of the user (required for security)
        query_text: Text to search in title and content
        category: Filter by category
        mood: Filter by mood
        tags: List of tags to filter by
        limit: Maximum number of results
        
    Returns:
        list: List of note IDs matching the search
    """
    try:
        ix = get_index()
        
        with ix.searcher() as searcher:
            # Build query
            queries = []
            
            # Always filter by user
            queries.append(Term('user_id', str(user_id)))
            
            # Text search in title and content
            if query_text:
                parser = MultifieldParser(['title', 'content'], schema=ix.schema)
                text_query = parser.parse(query_text)
                queries.append(text_query)
            
            # Category filter
            if category:
                queries.append(Term('category', category))
            
            # Mood filter
            if mood:
                queries.append(Term('mood', mood))
            
            # Tags filter
            if tags:
                tag_queries = [Term('tags', tag) for tag in tags]
                if len(tag_queries) > 1:
                    queries.append(Or(tag_queries))
                else:
                    queries.append(tag_queries[0])
            
            # Combine all queries
            if len(queries) > 1:
                final_query = And(queries)
            else:
                final_query = queries[0]
            
            # Execute search
            results = searcher.search(final_query, limit=limit)
            
            # Extract note IDs
            note_ids = [int(result['note_id']) for result in results]
            
            logger.info(f"Search returned {len(note_ids)} results")
            return note_ids
            
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        return []


def rebuild_index(notes_queryset):
    """
    Rebuild the entire search index from scratch.
    
    Args:
        notes_queryset: QuerySet of all notes to index
    """
    try:
        # Clear existing index
        index_dir = settings.WHOOSH_INDEX_DIR
        if os.path.exists(index_dir):
            import shutil
            shutil.rmtree(index_dir)
        
        # Create new index
        ix = get_index()
        writer = ix.writer()
        
        # Index all notes
        count = 0
        for note in notes_queryset:
            tags_str = ','.join([tag.name for tag in note.tags.all()])
            auto_tags = note.get_auto_tags_list()
            if auto_tags:
                tags_str += ',' + ','.join(auto_tags)
            
            writer.add_document(
                note_id=str(note.id),
                user_id=str(note.user.id),
                title=note.title or '',
                content=note.content,
                category=note.category or '',
                mood=note.mood or '',
                tags=tags_str,
                created_at=note.created_at,
            )
            count += 1
        
        writer.commit()
        logger.info(f"Rebuilt index with {count} notes")
        return count
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return 0

