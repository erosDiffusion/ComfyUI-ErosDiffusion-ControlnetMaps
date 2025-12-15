import sqlite3
import os
import time
import shutil

class MetadataManager:
    """Manages image metadata with schema versioning, favorites, and tags."""
    
    CURRENT_VERSION = 2
    
    def __init__(self, db_path):
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_or_migrate()
    
    def _ensure_db_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _get_version(self):
        """Get current schema version, returns 0 if not versioned."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if schema_version table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
                if not cursor.fetchone():
                    # Check if favorites table exists (V1 schema)
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='favorites'")
                    return 1 if cursor.fetchone() else 0
                
                cursor.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                return result[0] if result and result[0] else 0
        except Exception as e:
            print(f"[MetadataManager] Error getting version: {e}")
            return 0
    
    def _set_version(self, version):
        """Record a version as applied."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", 
                           (version, time.time()))
                conn.commit()
        except Exception as e:
            print(f"[MetadataManager] Error setting version: {e}")
    
    def _backup_db(self, version):
        """Create a backup before migration."""
        if os.path.exists(self.db_path):
            backup_path = f"{self.db_path}.backup.v{version}.{int(time.time())}"
            try:
                shutil.copy2(self.db_path, backup_path)
                print(f"[MetadataManager] Backed up DB to {backup_path}")
            except Exception as e:
                print(f"[MetadataManager] Backup failed: {e}")
    
    def _init_or_migrate(self):
        """Initialize DB or run migrations."""
        current_version = self._get_version()
        
        if current_version == 0:
            # Fresh install
            print(f"[MetadataManager] Creating fresh database at {self.db_path}")
            self._migrate_to_v1()
            self._migrate_to_v2()
        elif current_version < self.CURRENT_VERSION:
            # Need migration
            print(f"[MetadataManager] Migrating from v{current_version} to v{self.CURRENT_VERSION}")
            self._backup_db(current_version)
            for v in range(current_version + 1, self.CURRENT_VERSION + 1):
                print(f"[MetadataManager] Migrating to v{v}...")
                getattr(self, f'_migrate_to_v{v}')()
        else:
            print(f"[MetadataManager] Database is up to date (v{current_version})")

    
    def _migrate_to_v1(self):
        """V0 -> V1: Create initial schema with favorites."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create schema_version table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at REAL NOT NULL
                    )
                """)
                
                # Create favorites table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS favorites (
                        path TEXT PRIMARY KEY,
                        added_at REAL NOT NULL
                    )
                """)
                
                conn.commit()
                self._set_version(1)
                print("[MetadataManager] Migrated to v1")
        except Exception as e:
            print(f"[MetadataManager] Migration to v1 failed: {e}")
            raise
    
    def _migrate_to_v2(self):
        """V1 -> V2: Add tags tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create tags table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL
                    )
                """)
                
                # Create image_tags junction table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS image_tags (
                        image_path TEXT NOT NULL,
                        tag_id INTEGER NOT NULL,
                        added_at REAL NOT NULL,
                        PRIMARY KEY (image_path, tag_id),
                        FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_path ON image_tags(image_path)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id)")
                
                conn.commit()
                self._set_version(2)
                print("[MetadataManager] Migrated to v2")
        except Exception as e:
            print(f"[MetadataManager] Migration to v2 failed: {e}")
            raise
    
    # ===== Favorites API =====
    
    def toggle_favorite(self, path):
        """Toggles favorite status. Returns True if now favorite, False if removed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT path FROM favorites WHERE path = ?", (path,))
                exists = cursor.fetchone()
                
                if exists:
                    cursor.execute("DELETE FROM favorites WHERE path = ?", (path,))
                    conn.commit()
                    return False
                else:
                    cursor.execute("INSERT INTO favorites (path, added_at) VALUES (?, ?)", 
                                 (path, time.time()))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"[MetadataManager] Error toggling favorite: {e}")
            return False

    def is_favorite(self, path):
        """Check if path is favorited."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM favorites WHERE path = ?", (path,))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"[MetadataManager] Error checking favorite: {e}")
            return False

    def get_favorites(self):
        """Get all favorite paths."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT path FROM favorites ORDER BY added_at DESC")
                rows = cursor.fetchall()
                return [r[0] for r in rows]
        except Exception as e:
            print(f"[MetadataManager] Error listing favorites: {e}")
            return []
    
    # ===== Tags API =====
    
    def create_tag(self, name):
        """Create a new tag. Returns tag_id or None if exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
                conn.commit()
                cursor.execute("SELECT id FROM tags WHERE name = ?", (name,))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            print(f"[MetadataManager] Error creating tag: {e}")
            return None
    
    def get_or_create_tag(self, name):
        """Get tag_id, creating if necessary."""
        tag_id = self.get_tag_id(name)
        if tag_id is None:
            tag_id = self.create_tag(name)
        return tag_id
    
    def get_tag_id(self, name):
        """Get tag ID by name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM tags WHERE name = ?", (name,))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            print(f"[MetadataManager] Error getting tag ID: {e}")
            return None
    
    def add_tag_to_image(self, image_path, tag_name):
        """Add tag to image."""
        tag_id = self.get_or_create_tag(tag_name)
        if tag_id is None:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO image_tags (image_path, tag_id, added_at) 
                    VALUES (?, ?, ?)
                """, (image_path, tag_id, time.time()))
                conn.commit()
                return True
        except Exception as e:
            print(f"[MetadataManager] Error adding tag to image: {e}")
            return False
    
    def remove_tag_from_image(self, image_path, tag_name):
        """Remove tag from image."""
        tag_id = self.get_tag_id(tag_name)
        if tag_id is None:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM image_tags WHERE image_path = ? AND tag_id = ?", 
                           (image_path, tag_id))
                conn.commit()
                return True
        except Exception as e:
            print(f"[MetadataManager] Error removing tag from image: {e}")
            return False
    
    def get_tags_for_image(self, image_path):
        """Get all tags for an image."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.name FROM tags t
                    JOIN image_tags it ON t.id = it.tag_id
                    WHERE it.image_path = ?
                    ORDER BY t.name
                """, (image_path,))
                return [r[0] for r in cursor.fetchall()]
        except Exception as e:
            print(f"[MetadataManager] Error getting tags for image: {e}")
            return []
    
    def get_all_tags(self):
        """Get all tags with usage counts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.name, COUNT(it.image_path) as count
                    FROM tags t
                    LEFT JOIN image_tags it ON t.id = it.tag_id
                    GROUP BY t.id, t.name
                    ORDER BY t.name
                """)
                return [{"name": r[0], "count": r[1]} for r in cursor.fetchall()]
        except Exception as e:
            print(f"[MetadataManager] Error getting all tags: {e}")
            return []
    
    def delete_tag(self, tag_name):
        """Delete a tag (CASCADE removes all image associations)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM tags WHERE name = ?", (tag_name,))
                conn.commit()
                return True
        except Exception as e:
            print(f"[MetadataManager] Error deleting tag: {e}")
            return False
