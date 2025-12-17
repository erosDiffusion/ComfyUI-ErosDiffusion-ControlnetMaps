"""
Author: ErosDiffusion (EF)
Email: erosdiffusionai+controlnetmaps@gmail.com
Year: 2025
"""

import os
import torch
import numpy as np
import json
from PIL import Image, ImageOps
import folder_paths
from server import PromptServer
from aiohttp import web
from .metadata_manager import MetadataManager
import tempfile
import zipfile
import sqlite3
import shutil
from datetime import datetime
import time

# Config & Persistence
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "eros_config.json")
DB_PATH = os.path.join(NODE_DIR, "metadata.db")

metadata_manager = MetadataManager(DB_PATH)

# Ensure default maps directory exists (first-run friendly)
try:
    default_maps_dir = os.path.join(folder_paths.get_input_directory(), "maps")
    if not os.path.exists(default_maps_dir):
        os.makedirs(default_maps_dir, exist_ok=True)
        print(f"[CacheMap] Created default maps directory: {default_maps_dir}")
except Exception as e:
    print(f"[CacheMap] Warning: could not ensure default maps dir: {e}")

def load_map_types():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config.get("map_types", ["depth", "canny", "openpose", "lineart", "scribble", "softedge", "normal", "seg", "shuffle", "mediapipe_face", "custom"])
        except Exception as e:
            print(f"[CacheMap] Error loading config: {e}")
            return ["depth", "canny", "openpose", "lineart", "scribble", "softedge", "normal", "seg", "shuffle", "mediapipe_face", "custom"]
    return ["depth", "canny", "openpose", "lineart", "scribble", "softedge", "normal", "seg", "shuffle", "mediapipe_face", "custom"]


def _resolve_cache_root(raw_path: str) -> str:
    """Resolve a cache root path.

    For safety, export/import/reset are restricted to paths under the ComfyUI
    input directory.
    """
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    if not raw_path:
        target = default_maps_dir
    else:
        target = raw_path
        if not os.path.isabs(target):
            target = os.path.join(input_dir, target)

    target = os.path.abspath(target)
    try:
        if os.path.commonpath([input_dir, target]) != input_dir:
            raise ValueError("cache_path must be within ComfyUI input directory")
    except Exception:
        raise ValueError("Invalid cache_path")
    return target


def _safe_zip_members(zf: zipfile.ZipFile):
    for info in zf.infolist():
        name = info.filename
        if not name or name.endswith("/"):
            continue
        # Zip paths are always forward-slash separated.
        parts = [p for p in name.split("/") if p]
        if any(p == ".." for p in parts):
            continue
        if name.startswith("/") or name.startswith("\\"):
            continue
        yield info


def _sanitize_favorite_path(value: str, cache_root: str) -> str:
    if value is None:
        return ""
    s = str(value)
    s_strip = s.strip()
    if not s_strip:
        return ""

    # Normalize slashes for storage in DB.
    s_norm = s_strip.replace("\\", "/")

    # If absolute, try to make it relative to cache_root.
    looks_abs = (
        len(s_norm) >= 2 and s_norm[1] == ":"
    ) or s_norm.startswith("/") or s_norm.startswith("\\\\")
    if looks_abs:
        try:
            rel = os.path.relpath(s_strip, start=cache_root)
            rel = rel.replace("\\", "/")
            if not rel.startswith(".."):  # within cache_root
                return rel
        except Exception:
            pass
        parts = [p for p in s_norm.split("/") if p]
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1] if parts else s_norm

    return s_norm


def _sanitize_image_key(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    last = s.split("/")[-1]
    base = os.path.splitext(last)[0]
    return base


def _dump_sqlite_db_to_sql(db_path: str, cache_root: str) -> str:
    """Dump SQLite DB to SQL, sanitizing known path fields to be relative."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            lines = []
            for line in conn.iterdump():
                # Sanitize favorites.path and image_tags.image_path when present.
                # Handles forms like:
                # INSERT INTO "favorites" VALUES('path',123.0);
                # INSERT INTO favorites VALUES('path',123.0);
                # INSERT INTO "image_tags" VALUES('image',1,123.0);
                lowered = line.lower()
                if lowered.startswith("insert into \"favorites\" values(") or lowered.startswith(
                    "insert into favorites values("
                ):
                    try:
                        # first quoted string after VALUES(
                        start = line.find("VALUES(")
                        if start != -1:
                            frag = line[start + len("VALUES(") :]
                            if frag.startswith("'"):
                                end = frag.find("',")
                                if end != -1:
                                    raw = frag[1:end].replace("''", "'")
                                    sanitized = _sanitize_favorite_path(raw, cache_root)
                                    sanitized_sql = sanitized.replace("'", "''")
                                    line = (
                                        line[: start + len("VALUES(")]
                                        + "'"
                                        + sanitized_sql
                                        + frag[end:]
                                    )
                    except Exception:
                        pass
                elif lowered.startswith("insert into \"image_tags\" values(") or lowered.startswith(
                    "insert into image_tags values("
                ):
                    try:
                        start = line.find("VALUES(")
                        if start != -1:
                            frag = line[start + len("VALUES(") :]
                            if frag.startswith("'"):
                                end = frag.find("',")
                                if end != -1:
                                    raw = frag[1:end].replace("''", "'")
                                    sanitized = _sanitize_image_key(raw)
                                    sanitized_sql = sanitized.replace("'", "''")
                                    line = (
                                        line[: start + len("VALUES(")]
                                        + "'"
                                        + sanitized_sql
                                        + frag[end:]
                                    )
                    except Exception:
                        pass

                lines.append(line)
            return "\n".join(lines) + "\n"
        finally:
            conn.close()
    except Exception as e:
        return f"-- failed to dump {os.path.basename(db_path)}: {e}\n"


def _create_temp_db_from_sql(sql_text: str) -> str:
    fd, tmp_db = tempfile.mkstemp(prefix="eros_import_tmp_", suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(tmp_db)
    try:
        conn.executescript(sql_text)
        conn.commit()
    finally:
        conn.close()
    return tmp_db


def _merge_metadata_db_from_sql(target_db_path: str, sql_text: str, cache_root: str) -> dict:
    """Merge metadata.db content from a SQL dump, non-destructively."""
    stats = {
        "favorites_added": 0,
        "tags_added": 0,
        "image_tags_added": 0,
    }

    tmp_db = None
    try:
        # Ensure target schema exists
        try:
            MetadataManager(target_db_path)
        except Exception:
            pass

        tmp_db = _create_temp_db_from_sql(sql_text)

        src = sqlite3.connect(tmp_db)
        dst = sqlite3.connect(target_db_path)
        try:
            dst.execute("PRAGMA foreign_keys=ON")

            before = dst.total_changes
            try:
                rows = src.execute("SELECT path, added_at FROM favorites").fetchall()
                for path, added_at in rows:
                    sp = _sanitize_favorite_path(path, cache_root)
                    if not sp:
                        continue
                    dst.execute(
                        "INSERT OR IGNORE INTO favorites(path, added_at) VALUES (?, ?)",
                        (sp, float(added_at) if added_at is not None else time.time()),
                    )
            except Exception:
                pass
            dst.commit()
            stats["favorites_added"] = max(0, dst.total_changes - before)

            # Merge tags by name
            before = dst.total_changes
            try:
                tag_rows = src.execute("SELECT name FROM tags").fetchall()
                for (name,) in tag_rows:
                    if not name:
                        continue
                    dst.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (str(name),))
            except Exception:
                pass
            dst.commit()
            stats["tags_added"] = max(0, dst.total_changes - before)

            # Build dst name->id map
            name_to_id = {}
            try:
                for tid, name in dst.execute("SELECT id, name FROM tags").fetchall():
                    name_to_id[str(name)] = int(tid)
            except Exception:
                name_to_id = {}

            # Merge image_tags by (image_path, tag_name)
            before = dst.total_changes
            try:
                rows = src.execute(
                    """
                    SELECT it.image_path, t.name, it.added_at
                    FROM image_tags it
                    JOIN tags t ON t.id = it.tag_id
                    """
                ).fetchall()
                for image_path, tag_name, added_at in rows:
                    key = _sanitize_image_key(image_path)
                    if not key or not tag_name:
                        continue
                    dst_tid = name_to_id.get(str(tag_name))
                    if not dst_tid:
                        # Create missing tag then refetch id
                        try:
                            dst.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (str(tag_name),))
                            dst.commit()
                            r = dst.execute("SELECT id FROM tags WHERE name = ?", (str(tag_name),)).fetchone()
                            if r:
                                dst_tid = int(r[0])
                                name_to_id[str(tag_name)] = dst_tid
                        except Exception:
                            dst_tid = None
                    if not dst_tid:
                        continue

                    dst.execute(
                        """
                        INSERT OR IGNORE INTO image_tags(image_path, tag_id, added_at)
                        VALUES (?, ?, ?)
                        """,
                        (key, dst_tid, float(added_at) if added_at is not None else time.time()),
                    )
            except Exception:
                pass
            dst.commit()
            stats["image_tags_added"] = max(0, dst.total_changes - before)
        finally:
            try:
                src.close()
            except Exception:
                pass
            try:
                dst.close()
            except Exception:
                pass
    finally:
        try:
            if tmp_db and os.path.exists(tmp_db):
                os.remove(tmp_db)
        except Exception:
            pass

    return stats


def _merge_generic_db_from_sql(target_db_path: str, sql_text: str) -> dict:
    """Best-effort non-destructive merge for unknown DBs.

    - If DB doesn't exist: create it from SQL.
    - If it exists: create temp DB from SQL and INSERT OR IGNORE rows table-by-table.
    """
    stats = {"created": False, "rows_added": 0}

    if not os.path.exists(target_db_path):
        conn = sqlite3.connect(target_db_path)
        try:
            conn.executescript(sql_text)
            conn.commit()
            stats["created"] = True
        finally:
            conn.close()
        return stats

    tmp_db = None
    try:
        tmp_db = _create_temp_db_from_sql(sql_text)
        dst = sqlite3.connect(target_db_path)
        try:
            before = dst.total_changes
            dst.execute("ATTACH DATABASE ? AS src", (tmp_db,))
            try:
                tables = dst.execute(
                    "SELECT name, sql FROM src.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                for name, create_sql in tables:
                    if not name:
                        continue
                    # Ensure table exists
                    exists = dst.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,),
                    ).fetchone()
                    if not exists and create_sql:
                        try:
                            dst.execute(create_sql)
                        except Exception:
                            pass

                    try:
                        dst.execute(f"INSERT OR IGNORE INTO {name} SELECT * FROM src.{name}")
                    except Exception:
                        pass
                dst.commit()
            finally:
                try:
                    dst.execute("DETACH DATABASE src")
                except Exception:
                    pass
            stats["rows_added"] = max(0, dst.total_changes - before)
        finally:
            dst.close()
    finally:
        try:
            if tmp_db and os.path.exists(tmp_db):
                os.remove(tmp_db)
        except Exception:
            pass

    return stats


class CacheMapNode:
    @classmethod
    def INPUT_TYPES(s):
        default_path = os.path.join(folder_paths.get_input_directory(), "maps")
        input_config = {
            "required": {
                # Start with an empty default so the node uses the internal
                # `default_maps_dir` when `cache_path` is left blank by the user.
                "cache_path": ("STRING", {"default": "", "tooltip": "Root directory for the cache. Leave empty to use the Comfy input/maps folder."}),
                "filename": ("STRING", {"forceInput": True, "tooltip": "The unique identifier (base filename) for the map. Use 'Load Image ErosDiffusion' to extract this from a source image."}),
                "map_type": (["auto"] + load_map_types() + ["browser"], {"default": "auto", "tooltip": "The type of map to handle. 'browser' is a pure pass-through."}),
                "save_if_new": ("BOOLEAN", {"default": True, "tooltip": "If True, saves the generated map to the cache directory if it wasn't found."}),
                "force_generation": ("BOOLEAN", {"default": True, "tooltip": "If True, ignores existing cache and forces regeneration + overwrite."}),
                "generate_all": ("BOOLEAN", {"default": False, "tooltip": "If True, triggers ALL connected preprocessors and saves their maps (respecting force_generation)."}),
            },
            "optional": {
                "tags": ("STRING", {"default": "", "multiline": False}),
                "source_browser": ("IMAGE", {"lazy": True, "tooltip": "Lazy input. Connect CacheMap Browser here. Passes through the image without saving/modifying."}),
                "source_original": ("IMAGE", {"lazy": True, "tooltip": "Lazy input. Connect the Original Image here. It will be saved to 'original' folder for overlay in browser."}),
            }
        }
        
        # Dynamically add optional inputs based on config
        for mt in load_map_types():
            input_config["optional"][f"source_{mt}"] = ("IMAGE", {"lazy": True, "tooltip": f"Lazy input. Connect your {mt} Preprocessor here. Only runs if cache misses."})
            
        return input_config

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("map",)
    FUNCTION = "process"
    CATEGORY = "ErosDiffusion"
    DESCRIPTION = "Smart caching node for controlnet maps. Checks for existing maps in the cache directory to skip expensive generation. Supports 'auto' mode to automatically detect map types from connections. Use 'Generate All' to batch process all connected inputs."

    def _get_map_types(self):
         return load_map_types()

    def _is_connected_input(self, val):
        """Return True if `val` looks like a connected image input from ComfyUI.
        Accepts either a torch.Tensor or a (tensor, ...) tuple/list as produced
        by image outputs. This is used during lazy-checking to avoid asking the
        scheduler to request inputs that are not actually connected.
        """
        if val is None:
            return False
        if isinstance(val, torch.Tensor):
            return True
        if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], torch.Tensor):
            return True
        return False

    def _get_cache_file_paths(self, cache_path, map_type, filename):
        # Normalize cache_path: use default maps dir when empty, and
        # resolve relative paths against Comfy input directory.
        if not cache_path:
            cache_path = default_maps_dir
        elif not os.path.isabs(cache_path):
            cache_path = os.path.join(folder_paths.get_input_directory(), cache_path)

        basename = os.path.splitext(os.path.basename(filename))[0]
        target_dir = os.path.join(cache_path, map_type)
        extensions = [".png", ".jpg", ".jpeg", ".webp"]
        file_paths = [os.path.join(target_dir, basename + ext) for ext in extensions]
        return target_dir, file_paths

    def _check_exists(self, file_paths):
        for path in file_paths:
            if os.path.exists(path):
                return path
        return None

    def check_lazy_status(self, cache_path, filename, map_type, save_if_new, force_generation, generate_all, **kwargs):
        if filename is None:
             return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all"] + [f"source_{t}" for t in self._get_map_types()] + ["source_browser", "source_original"]

        # Browser Passthrough Mode
        if map_type == "browser":
            # Always request browser input, ignore cache checks
            return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all", "source_browser"]

        if generate_all:
            # Request ALL inputs to ensure they run
            # print(f"[CacheMap] Generate All Enabled. Requesting all connected inputs.")
            return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all"] + [f"source_{t}" for t in self._get_map_types()] + ["source_original"]

        if force_generation:
            # print(f"[CacheMap] Force Generation Enabled. Requesting all inputs to regenerate map for {filename}.")
            if map_type == "auto":
                 return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all"] + [f"source_{t}" for t in self._get_map_types()] + ["source_original"]
            else:
                 return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all", f"source_{map_type}"]

        if map_type == "auto":
            # Prefer any truly connected source_<type> inputs first (so the
            # scheduler only requests the connected one). Use the class helper
            # `_is_connected_input` to detect real connected image values.
            for type_check in self._get_map_types():
                key = f"source_{type_check}"
                try:
                    val = kwargs.get(key, None)
                    if self._is_connected_input(val):
                        return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", key]
                except Exception:
                    pass

            # Scan filesystem for an existing cached map first
            for type_check in self._get_map_types():
                _, file_paths = self._get_cache_file_paths(cache_path, type_check, filename)
                if self._check_exists(file_paths):
                    return ["cache_path", "filename", "map_type", "save_if_new", "force_generation"]

            # Not found: Request ALL inputs so the connected one runs
            print(f"[CacheMap] Auto-Miss: No map found. Requesting all inputs to trigger generation.")
            return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all"] + [f"source_{t}" for t in self._get_map_types()] + ["source_original"]

        else:
            # Specific type check
            needed_input = f"source_{map_type}"
            _, file_paths = self._get_cache_file_paths(cache_path, map_type, filename)
            
            if self._check_exists(file_paths):
                # print(f"[CacheMap] Cache HIT for {map_type} map of {filename}. Skipping generation.")
                return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all"]
            else:
                print(f"[CacheMap] Cache MISS for {map_type} map of {filename}. Requesting generation.")
                return ["cache_path", "filename", "map_type", "save_if_new", "force_generation", "generate_all", needed_input, "source_original"]

    def process(self, cache_path, filename, map_type, save_if_new, force_generation, generate_all, **kwargs):
        
        # Extract tags parameter
        tags_str = kwargs.get("tags", "")
        print(f"[CacheMap] process() called - filename='{filename}', map_type='{map_type}', tags='{tags_str}', force_generation={force_generation}, generate_all={generate_all}")
        
        # Normalize cache_path early so all subsequent operations use a
        # resolved absolute path. If empty, default to the Comfy input/maps dir.
        if not cache_path:
            cache_path = default_maps_dir
        elif not os.path.isabs(cache_path):
            cache_path = os.path.join(folder_paths.get_input_directory(), cache_path)

        # Browser Passthrough
        if map_type == "browser":
            img = kwargs.get("source_browser")
            if img is None:
                print("[CacheMap] Browser mode selected but no input connected/loaded.")
                return (torch.zeros((1, 512, 512, 3)),)
            return (img,)

        # Helper function to save tags
        # If notify_on_complete is None we send frontend notifications immediately.
        # If notify_on_complete is a set, we defer notifications and add basenames to it.
        def save_tags_for_image(filename_to_tag, tags_string):
            """Parse and save tags to database for the given filename."""
            print(f"[CacheMap] save_tags_for_image called with filename='{filename_to_tag}', tags='{tags_string}'")
            
            if not tags_string or not tags_string.strip():
                print(f"[CacheMap] No tags to process (empty or whitespace-only)")
                return
            
            # Get basename without extension for database key
            basename = os.path.splitext(os.path.basename(filename_to_tag))[0]
            print(f"[CacheMap] Extracted basename: '{basename}'")
            
            # Parse comma-separated tags, trim whitespace, and remove duplicates
            tag_list = []
            seen_tags = set()
            
            print(f"[CacheMap] Parsing tags from string: '{tags_string}'")
            for idx, tag in enumerate(tags_string.split(',')):
                print(f"[CacheMap]   Tag {idx}: raw='{tag}'")
                
                # Trim leading/trailing whitespace
                tag = tag.strip()
                print(f"[CacheMap]   Tag {idx}: trimmed='{tag}'")
                
                # Skip empty tags
                if not tag:
                    print(f"[CacheMap]   Tag {idx}: SKIPPED (empty after trim)")
                    continue
                
                # Normalize to lowercase for duplicate detection (but preserve original case for storage)
                tag_lower = tag.lower()
                
                # Skip duplicates (case-insensitive)
                if tag_lower in seen_tags:
                    print(f"[CacheMap]   Tag {idx}: SKIPPED (duplicate of '{tag}')")
                    continue
                
                seen_tags.add(tag_lower)
                tag_list.append(tag)
                print(f"[CacheMap]   Tag {idx}: ADDED to list")
            
            if tag_list:
                print(f"[CacheMap] Processing {len(tag_list)} unique tag(s) for '{basename}': {tag_list}")
                for tag in tag_list:
                    print(f"[CacheMap] Calling metadata_manager.add_tag_to_image('{basename}', '{tag}')")
                    success = metadata_manager.add_tag_to_image(basename, tag)
                    if success:
                        print(f"[CacheMap] ✓ Successfully added tag '{tag}' to '{basename}'")
                    else:
                        print(f"[CacheMap] ⚠ Tag '{tag}' already exists for '{basename}' (skipped)")

                # Verify tags were saved
                print(f"[CacheMap] Verifying saved tags for '{basename}'...")
                saved_tags = metadata_manager.get_tags_for_image(basename)
                print(f"[CacheMap] Tags in database for '{basename}': {saved_tags}")

                # By default, notify frontend immediately. If defer is enabled
                # the caller may add this basename to `notify_on_complete` set
                # and notifications will be sent once processing finishes.
                if notify_on_complete is None:
                    print(f"[CacheMap] Sending tag update notification to frontend for '{basename}'")
                    PromptServer.instance.send_sync("eros.tags.updated", {
                        "basename": basename,
                        "tags": saved_tags
                    })
                else:
                    # Caller will handle notifying after batch operations
                    notify_on_complete.add(basename)
            else:
                print(f"[CacheMap] No valid tags to process after filtering")


        # Use the class helper `_is_connected_input` instead of a local helper.


        # Generate All Logic
        # Set up a defer-notify set when doing batch generation so we can
        # send frontend updates only after all maps are processed.
        notify_on_complete = None
        # Collect saved map info during this run so we can emit a single
        # `eros.map.saved` notification once at the end (avoids multiple refreshes)
        saved_maps = []
        if generate_all:
            notify_on_complete = set()
            print(f"[CacheMap] Processing 'Generate All' for {filename}...")
            for type_check in self._get_map_types():
                if type_check == "custom":
                    continue

                source_img = kwargs.get(f"source_{type_check}")
                if self._is_connected_input(source_img):
                    # Check if we should save
                    target_dir, file_paths = self._get_cache_file_paths(cache_path, type_check, filename)
                    exists = self._check_exists(file_paths)

                    if force_generation or not exists:
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir, exist_ok=True)
                        save_path = os.path.join(target_dir, os.path.splitext(os.path.basename(filename))[0] + ".png")

                        img_tensor = source_img[0]
                        img_array = (img_tensor * 255.0).cpu().numpy().astype(np.uint8)
                        img = Image.fromarray(img_array)
                        img.save(save_path)
                        print(f"[CacheMap] Generate All: Saved {type_check} -> {save_path}")

                        # Save tags when generating/regenerating; defer frontend notify
                        save_tags_for_image(filename, tags_str)
                        # Record saved map for end-of-run notification
                        try:
                            saved_maps.append({"basename": os.path.splitext(os.path.basename(filename))[0], "type": type_check, "path": save_path})
                        except Exception:
                            pass
                    else:
                        print(f"[CacheMap] Generate All: Skipped {type_check} (Exists)")

            # Also handle source_original during Generate All
            orig_img = kwargs.get("source_original")
            if orig_img is not None:
                target_dir = os.path.join(cache_path, "original")
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)

                save_path = os.path.join(target_dir, os.path.splitext(os.path.basename(filename))[0] + ".png")
                if force_generation or not os.path.exists(save_path):
                    img_tensor = orig_img[0]
                    img_array = (img_tensor * 255.0).cpu().numpy().astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img.save(save_path)
                    print(f"[CacheMap] Generate All: Saved original -> {save_path}")

                    # Save tags for original image (defer notify)
                    save_tags_for_image(filename, tags_str)
                    try:
                        saved_maps.append({"basename": os.path.splitext(os.path.basename(filename))[0], "type": "original", "path": save_path})
                    except Exception:
                        pass

        # Process Single Flow (saving source_original if present)
        # We check this every run if connected, to ensure overlay availability
        if kwargs.get("source_original") is not None and not generate_all:
            orig_img = kwargs.get("source_original")
            target_dir = os.path.join(cache_path, "original")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            save_path = os.path.join(target_dir, os.path.splitext(os.path.basename(filename))[0] + ".png")
            
            # Only save if new or forced
            if save_if_new or force_generation:
                if force_generation or not os.path.exists(save_path):
                     img_tensor = orig_img[0]
                     img_array = (img_tensor * 255.0).cpu().numpy().astype(np.uint8)
                     img = Image.fromarray(img_array)
                     img.save(save_path)
                     print(f"[CacheMap] Saved original image for overlay -> {save_path}")
                     
                     # Save tags for original image
                     save_tags_for_image(filename, tags_str)
                     try:
                         saved_maps.append({"basename": os.path.splitext(os.path.basename(filename))[0], "type": "original", "path": save_path})
                     except Exception:
                         pass

        # Resolve 'auto' to actual type if possible (for saving) or just load existing
        target_type = map_type
        existing_file = None
        
        if not force_generation:
            if map_type == "auto":
                 # Try to find existing first
                 for type_check in self._get_map_types():
                    _, file_paths = self._get_cache_file_paths(cache_path, type_check, filename)
                    found = self._check_exists(file_paths)
                    if found:
                        existing_file = found
                        target_type = type_check
                        break
            else:
                _, file_paths = self._get_cache_file_paths(cache_path, map_type, filename)
                existing_file = self._check_exists(file_paths)

        if existing_file and not force_generation:
            img = Image.open(existing_file)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            output_image = np.array(img).astype(np.float32) / 255.0
            output_image = torch.from_numpy(output_image)[None,]
            return (output_image,)
        
        # Cache Miss OR Forced Generation
        generated_map = None
        
        if map_type == "auto":
            # Find the first non-None input
            for type_check in self._get_map_types():
                key = f"source_{type_check}"
                val = kwargs.get(key)
                if self._is_connected_input(val):
                    generated_map = val
                    target_type = type_check
                    break
        else:
            generated_map = kwargs.get(f"source_{map_type}")
            target_type = map_type

        if generated_map is None:
            print(f"[CacheMap] Error: Generation required (Force: {force_generation}) but no input provided (Mode: {map_type}).")
            return (torch.zeros((1, 512, 512, 3)),)

        if save_if_new or force_generation:
            target_dir, _ = self._get_cache_file_paths(cache_path, target_type, filename)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            save_path = os.path.join(target_dir, os.path.splitext(os.path.basename(filename))[0] + ".png")
            
            img_tensor = generated_map[0] 
            img_array = (img_tensor * 255.0).cpu().numpy().astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(save_path)
            print(f"[CacheMap] Saved {'(FORCED) ' if force_generation else ''}{target_type} map to {save_path}")

            # Save tags when generating/regenerating
            save_tags_for_image(filename, tags_str)
            try:
                saved_maps.append({"basename": os.path.splitext(os.path.basename(filename))[0], "type": target_type, "path": save_path})
            except Exception:
                pass

        # After all processing, if we deferred notifications for batch ops,
        # send a single update per basename to the frontend so it only refreshes once.
        if notify_on_complete:
            for basename in list(notify_on_complete):
                try:
                    saved_tags = metadata_manager.get_tags_for_image(basename)
                    print(f"[CacheMap] Sending batch tag update for '{basename}': {saved_tags}")
                    PromptServer.instance.send_sync("eros.tags.updated", {"basename": basename, "tags": saved_tags})
                except Exception as e:
                    print(f"[CacheMap] Error sending batch tag update for '{basename}': {e}")

        # Notify frontend once about all saved maps in this run so the browser
        # can refresh a single time and reapply filters.
        if saved_maps:
            try:
                PromptServer.instance.send_sync("eros.map.saved", {"saved": saved_maps})
            except Exception:
                pass

        return (generated_map,)

class CacheMapBrowserNode:
    @classmethod
    def INPUT_TYPES(s):
        default_path = os.path.join(folder_paths.get_input_directory(), "maps")
        return {
            "required": {
                # Use empty default so the browser node also falls back to the
                # internal default maps directory when nothing is entered.
                "cache_path": ("STRING", {"default": "", "tooltip": "Root directory for the cache. Leave empty to use the Comfy input/maps folder."}),
            },
            "optional": {
                "extra_path": ("STRING", {"default": "", "tooltip": "Additional path to browse."}),
                 # Filename widget will be populated by JS
                "filename": ("STRING", {"default": "", "tooltip": "Selected filename (relative to cache/extra path)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "ErosDiffusion"
    DESCRIPTION = "Visual browser for cache maps. Adds a 'Open Browser' button to browse and select maps from the sidebar."

    def load_image(self, cache_path, filename, extra_path=None):
        # Determine full path
        # filename is relative e.g. "depth/my_file.png"
        
        image_path = None

        # Defensive: if filename is empty or None, return empty tensors
        if not filename or not str(filename).strip():
            print(f"[CacheMapBrowser] load_image called with empty filename (cache_path={cache_path})")
            return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)))

        # Resolve base cache directory: use provided cache_path or default maps dir
        base_cache = cache_path or default_maps_dir
        # If relative, resolve against Comfy input directory
        if not os.path.isabs(base_cache):
            base_cache = os.path.join(folder_paths.get_input_directory(), base_cache)

        # Try direct join (handles prefixed 'type/filename' entries)
        p1 = os.path.join(base_cache, filename)
        if os.path.exists(p1) and os.path.isfile(p1):
            image_path = p1
        # Try extra_path if provided
        elif extra_path:
            ep = extra_path or ""
            if not os.path.isabs(ep):
                ep = os.path.join(folder_paths.get_input_directory(), ep)
            p2 = os.path.join(ep, filename)
            if os.path.exists(p2) and os.path.isfile(p2):
                image_path = p2
        else:
            # As a fallback, if filename contains a prefix like 'type/name', also try splitting
            if "/" in filename:
                parts = filename.split("/")
                sub = parts[0]
                name_only = "/".join(parts[1:])
                alt = os.path.join(base_cache, sub, name_only)
                if os.path.exists(alt) and os.path.isfile(alt):
                    image_path = alt
        
        if image_path is None:
            #  print(f"[CacheMapBrowser] File not found: {filename}")
             # Return empty
             return (torch.zeros((1, 512, 512, 3)), torch.zeros((1, 512, 512)))

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((1, img.height, img.width), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        output_image = np.array(img).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image)[None,]
        
        return (output_image, mask)

NODE_CLASS_MAPPINGS = {
    "CacheMapNode": CacheMapNode,
    "CacheMapBrowserNode": CacheMapBrowserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheMapNode": "ControlNet Map Cache (ErosDiffusion)",
    "CacheMapBrowserNode": "ControlNet Map Browser (ErosDiffusion)"
}


# ================= API Routes =================

@PromptServer.instance.routes.get("/eros/cache/fetch_dirs")
async def fetch_dirs(request):
    # Use default maps dir when no path provided or path is empty
    target_path = request.rel_url.query.get("path", "")
    if not target_path:
        target_path = default_maps_dir
    else:
        # Accept relative paths: resolve against Comfy input directory
        if not os.path.isabs(target_path):
            target_path = os.path.join(folder_paths.get_input_directory(), target_path)
    if not os.path.exists(target_path):
         return web.json_response({"dirs": []})
    
    dirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
    return web.json_response({"dirs": sorted(dirs)})

@PromptServer.instance.routes.get("/eros/cache/fetch_files")
async def fetch_files(request):
    # Use default maps dir when no path provided or path is empty
    target_path = request.rel_url.query.get("path", "")
    if not target_path:
        target_path = default_maps_dir
    else:
        if not os.path.isabs(target_path):
            target_path = os.path.join(folder_paths.get_input_directory(), target_path)

    # Optional subfolder (e.g. map_type)
    subfolder = request.rel_url.query.get("subfolder", "")
    # Sanitize accidental object stringification from client-side
    if isinstance(subfolder, str) and "[object Object]" in subfolder:
        subfolder = ""

    search_path = os.path.join(target_path, subfolder) if subfolder else target_path

    if not os.path.exists(search_path):
         return web.json_response({"files": []})

    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = []
    
    try:
        for f in os.listdir(search_path):
            if os.path.isfile(os.path.join(search_path, f)):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_ext:
                    files.append(f)
    except Exception as e:
         return web.json_response({"error": str(e)}, status=500)
         
    return web.json_response({"files": sorted(files)})

@PromptServer.instance.routes.get("/eros/cache/view_image")
async def view_image(request):
    filename = request.rel_url.query.get("filename")
    if not filename:
        return web.Response(status=400)

    target_path = request.rel_url.query.get("path", "")
    if not target_path:
        target_path = default_maps_dir
    else:
        if not os.path.isabs(target_path):
            target_path = os.path.join(folder_paths.get_input_directory(), target_path)

    subfolder = request.rel_url.query.get("subfolder", "")
    # sanitize
    if isinstance(subfolder, str) and "[object Object]" in subfolder:
        subfolder = ""

    full_path = os.path.join(target_path, subfolder, filename) if subfolder else os.path.join(target_path, filename)

    if not os.path.exists(full_path):
        return web.Response(status=404)

    return web.FileResponse(full_path)

# ================= Favorites API =================

@PromptServer.instance.routes.post("/eros/favorites/toggle")
async def toggle_favorite(request):
    try:
        data = await request.json()
        path = data.get("path")
        if not path:
            return web.json_response({"error": "Missing path"}, status=400)
        
        is_fav = metadata_manager.toggle_favorite(path)
        return web.json_response({"path": path, "is_favorite": is_fav})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/eros/favorites/list")
async def list_favorites(request):
    try:
        favs = metadata_manager.get_favorites()
        return web.json_response({"favorites": favs})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

@PromptServer.instance.routes.post("/eros/tags/auto_tag")
async def auto_tag_image(request):
    """
    Auto-tag an image using AI/LLM (placeholder implementation).
    Expects JSON: {"path": "image_basename"}
    Returns: {"tags": ["tag1", "tag2", ...]}
    """
    try:
        data = await request.json()
        image_path = data.get("path", "")
        
        # TODO: Implement actual LLM-based tagging
        # For now, return mock tags
        mock_tags = ["test", "automatic", "tagging"]
        
        return web.json_response({"success": True, "tags": mock_tags})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

# ================= Tags API =================

@PromptServer.instance.routes.post("/eros/tags/create")
async def create_tag(request):
    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return web.json_response({"error": "Missing tag name"}, status=400)
        
        tag_id = metadata_manager.create_tag(name)
        if tag_id:
            return web.json_response({"tag_id": tag_id, "name": name})
        else:
            return web.json_response({"error": "Tag already exists or creation failed"}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/eros/tags/add_to_image")
async def add_tag_to_image(request):
    try:
        data = await request.json()
        path = data.get("path")
        tag = data.get("tag")
        if not path or not tag:
            return web.json_response({"error": "Missing path or tag"}, status=400)
        
        success = metadata_manager.add_tag_to_image(path, tag)
        return web.json_response({"success": success, "path": path, "tag": tag})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/eros/tags/remove_from_image")
async def remove_tag_from_image(request):
    try:
        data = await request.json()
        path = data.get("path")
        tag = data.get("tag")
        if not path or not tag:
            return web.json_response({"error": "Missing path or tag"}, status=400)
        
        success = metadata_manager.remove_tag_from_image(path, tag)
        return web.json_response({"success": success, "path": path, "tag": tag})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/eros/tags/list")
async def list_tags(request):
    try:
        tags = metadata_manager.get_all_tags()
        return web.json_response({"tags": tags})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/eros/tags/for_image")
async def get_tags_for_image(request):
    try:
        path = request.rel_url.query.get("path")
        if not path:
            return web.json_response({"error": "Missing path"}, status=400)
        
        tags = metadata_manager.get_tags_for_image(path)
        return web.json_response({"path": path, "tags": tags})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.delete("/eros/tags/delete")
async def delete_tag(request):
    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return web.json_response({"error": "Missing tag name"}, status=400)
        
        success = metadata_manager.delete_tag(name)
        return web.json_response({"success": success, "name": name})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/eros/cache/delete_map")
async def delete_map(request):
    """Delete a cached map file. JSON body expects:
       { "cache_path": <optional>, "subfolder": <optional>, "filename": <required>, "basename": <optional>, "delete_all": <bool> }

    If `delete_all` is true, all files matching the basename across subfolders
    under the cache path will be removed. Otherwise only the provided
    `subfolder/filename` (or filename relative to cache_path) is removed.
    After deletion, any tag associations for the basename will be removed
    from the metadata DB (tags themselves are preserved).
    """
    try:
        data = await request.json()
        cache_path = data.get("cache_path", "")
        subfolder = data.get("subfolder", "")
        filename = data.get("filename")
        basename = data.get("basename")
        delete_all = bool(data.get("delete_all", False))

        if not filename and not basename:
            return web.json_response({"error": "Missing filename or basename"}, status=400)

        # Resolve base cache path
        if not cache_path:
            target_path = default_maps_dir
        else:
            target_path = cache_path
            if not os.path.isabs(target_path):
                target_path = os.path.join(folder_paths.get_input_directory(), target_path)

        deleted = []

        # Determine basename if missing
        if not basename and filename:
            basename = os.path.splitext(os.path.basename(filename))[0]

        # If delete_all, scan subdirectories
        if delete_all:
            if not os.path.exists(target_path):
                return web.json_response({"deleted": deleted})
            for root, dirs, files in os.walk(target_path):
                for f in list(files):
                    if os.path.splitext(f)[0] == basename:
                        p = os.path.join(root, f)
                        try:
                            os.remove(p)
                            deleted.append(os.path.relpath(p, start=target_path))
                        except Exception:
                            pass
        else:
            # Delete the specific file path
            # Accept either a subfolder+filename or filename that may already include subfolder
            if subfolder and filename:
                full = os.path.join(target_path, subfolder, filename)
            elif filename and "/" in filename:
                full = os.path.join(target_path, filename)
            elif filename:
                # try currentTab-like resolution: search for file in subfolders
                full = os.path.join(target_path, filename)

            if os.path.exists(full) and os.path.isfile(full):
                try:
                    os.remove(full)
                    deleted.append(os.path.relpath(full, start=target_path))
                except Exception:
                    pass

        # Remove tag associations for this basename
        removed_count = 0
        if basename:
            try:
                removed_count = metadata_manager.remove_tags_for_image(basename)
            except Exception:
                removed_count = 0

        # Notify frontend(s)
        try:
            PromptServer.instance.send_sync("eros.image.deleted", {"basename": basename, "deleted": deleted})
            # Also send an empty tags update so clients know tags for this basename are gone
            PromptServer.instance.send_sync("eros.tags.updated", {"basename": basename, "tags": []})
        except Exception:
            pass

        return web.json_response({"success": True, "deleted": deleted, "removed_tag_links": removed_count})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/eros/cache/export_zip")
async def export_zip(request):
    """Export cache folder + sqlite dbs as a zip.

    Query params:
      - path: optional cache root (relative to input dir). Defaults to input/maps.
    """
    try:
        raw_path = request.rel_url.query.get("path", "")
        cache_root = _resolve_cache_root(raw_path)

        if not os.path.exists(cache_root):
            os.makedirs(cache_root, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"eros_maps_export_{ts}.zip"

        fd, tmp_path = tempfile.mkstemp(prefix="eros_maps_export_", suffix=".zip")
        os.close(fd)

        input_dir = os.path.abspath(folder_paths.get_input_directory())
        cache_rel = os.path.relpath(cache_root, start=input_dir).replace("\\", "/")

        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # maps/*
                for root, dirs, files in os.walk(cache_root):
                    for f in files:
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, start=cache_root).replace("\\", "/")
                        zf.write(full, arcname=f"maps/{rel}")

                # db/*.sql
                db_files = [
                    os.path.join(NODE_DIR, f)
                    for f in os.listdir(NODE_DIR)
                    if f.lower().endswith(".db") and os.path.isfile(os.path.join(NODE_DIR, f))
                ]
                # Always include the metadata.db dump even if the file is missing
                # (fresh install) so imports can still recreate it.
                if DB_PATH not in db_files:
                    db_files.append(DB_PATH)

                for dbp in db_files:
                    db_name = os.path.basename(dbp)
                    sql_text = _dump_sqlite_db_to_sql(dbp, cache_root)
                    zf.writestr(f"db/{db_name}.sql", sql_text)

                # manifest
                manifest = {
                    "version": 1,
                    "cache_root": cache_rel,
                    "exported_at": ts,
                    "db": [os.path.basename(p) + ".sql" for p in db_files],
                }
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            with open(tmp_path, "rb") as rf:
                body = rf.read()

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            return web.Response(
                body=body,
                headers={
                    "Content-Type": "application/zip",
                    "Content-Disposition": f'attachment; filename="{out_name}"',
                },
            )
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
    except ValueError as ve:
        return web.json_response({"error": str(ve)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/eros/cache/import_zip")
async def import_zip(request):
    """Import a zip created by /eros/cache/export_zip.

    Query params:
      - path: optional cache root (relative to input dir). Defaults to input/maps.

    Non-destructive merge: keeps existing local cache + DBs and adds missing data.
    """
    global metadata_manager
    try:
        raw_path = request.rel_url.query.get("path", "")
        cache_root = _resolve_cache_root(raw_path)
        os.makedirs(cache_root, exist_ok=True)

        reader = await request.multipart()
        part = None
        while True:
            p = await reader.next()
            if p is None:
                break
            if p.name in ("file", "archive"):
                part = p
                break

        if not part:
            return web.json_response({"error": "Missing upload field 'file'"}, status=400)

        fd, tmp_zip = tempfile.mkstemp(prefix="eros_maps_import_", suffix=".zip")
        os.close(fd)
        try:
            with open(tmp_zip, "wb") as f:
                while True:
                    chunk = await part.read_chunk(size=1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)

            imported_files = 0
            skipped_files = 0
            imported_dbs = 0
            db_rows_added = 0
            metadata_merge = {"favorites_added": 0, "tags_added": 0, "image_tags_added": 0}

            with zipfile.ZipFile(tmp_zip, "r") as zf:
                sql_by_db = {}
                for info in _safe_zip_members(zf):
                    name = info.filename
                    if name.startswith("maps/"):
                        rel = name[len("maps/") :]
                        if not rel:
                            continue
                        # Prevent zip slip
                        rel_parts = [p for p in rel.split("/") if p]
                        if any(p == ".." for p in rel_parts):
                            continue
                        dest = os.path.abspath(os.path.join(cache_root, *rel_parts))
                        if os.path.commonpath([cache_root, dest]) != cache_root:
                            continue
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        if os.path.exists(dest):
                            skipped_files += 1
                        else:
                            with zf.open(info, "r") as src, open(dest, "wb") as out:
                                shutil.copyfileobj(src, out)
                            imported_files += 1
                    elif name.startswith("db/") and name.endswith(".sql"):
                        db_name = os.path.basename(name[:-4])  # strip .sql
                        try:
                            sql_text = zf.read(info).decode("utf-8", errors="replace")
                            sql_by_db[db_name] = sql_text
                        except Exception:
                            pass

                # Restore DBs from SQL
                for db_name, sql_text in sql_by_db.items():
                    # db_name should be like metadata.db
                    if not db_name.lower().endswith(".db"):
                        continue
                    db_path = os.path.join(NODE_DIR, db_name)
                    try:
                        if os.path.abspath(db_path) == os.path.abspath(DB_PATH):
                            m = _merge_metadata_db_from_sql(db_path, sql_text, cache_root)
                            metadata_merge["favorites_added"] += int(m.get("favorites_added", 0) or 0)
                            metadata_merge["tags_added"] += int(m.get("tags_added", 0) or 0)
                            metadata_merge["image_tags_added"] += int(m.get("image_tags_added", 0) or 0)
                            db_rows_added += (
                                int(m.get("favorites_added", 0) or 0)
                                + int(m.get("tags_added", 0) or 0)
                                + int(m.get("image_tags_added", 0) or 0)
                            )
                            imported_dbs += 1
                        else:
                            g = _merge_generic_db_from_sql(db_path, sql_text)
                            db_rows_added += int(g.get("rows_added", 0) or 0)
                            imported_dbs += 1
                    except Exception:
                        # best-effort; continue
                        pass

            # Re-init metadata manager to ensure schema is available post-import
            try:
                metadata_manager = MetadataManager(DB_PATH)
            except Exception:
                pass

            try:
                PromptServer.instance.send_sync(
                    "eros.cache.imported",
                    {
                        "imported_files": imported_files,
                        "skipped_files": skipped_files,
                        "imported_dbs": imported_dbs,
                        "db_rows_added": db_rows_added,
                        "metadata_merge": metadata_merge,
                    },
                )
            except Exception:
                pass

            return web.json_response(
                {
                    "success": True,
                    "imported_files": imported_files,
                    "skipped_files": skipped_files,
                    "imported_dbs": imported_dbs,
                    "db_rows_added": db_rows_added,
                    "metadata_merge": metadata_merge,
                }
            )
        finally:
            try:
                if os.path.exists(tmp_zip):
                    os.remove(tmp_zip)
            except Exception:
                pass
    except ValueError as ve:
        return web.json_response({"error": str(ve)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/eros/cache/reset")
async def reset_cache(request):
    """Delete all cached maps under cache root and wipe metadata.

    JSON body:
      - path: optional cache root
      - wipe_other_dbs: bool (if true, also deletes other .db files in NODE_DIR)
    """
    global metadata_manager
    try:
        data = await request.json()
    except Exception:
        data = {}

    try:
        raw_path = data.get("path", "") if isinstance(data, dict) else ""
        # Default to full reset (most reliable) if caller doesn't specify.
        wipe_other_dbs = bool(data.get("wipe_other_dbs", True)) if isinstance(data, dict) else True
        cache_root = _resolve_cache_root(raw_path)
        os.makedirs(cache_root, exist_ok=True)

        removed_files = 0
        for name in os.listdir(cache_root):
            p = os.path.join(cache_root, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed_files += 1
            except Exception:
                pass

        def _try_delete_db_file(db_path: str) -> bool:
            ok = False
            try:
                if os.path.isfile(db_path):
                    os.remove(db_path)
                    ok = True
            except Exception:
                ok = False
            # Also try to remove sqlite sidecar files
            for suffix in ("-wal", "-shm", "-journal"):
                try:
                    side = db_path + suffix
                    if os.path.isfile(side):
                        os.remove(side)
                except Exception:
                    pass
            return ok

        def _clear_all_user_tables(db_path: str) -> bool:
            """Fallback for when a DB file can't be deleted (Windows locks).

            Deletes rows from all non-sqlite_* tables. Best-effort.
            """
            try:
                conn = sqlite3.connect(db_path, timeout=2.0)
            except Exception:
                return False
            try:
                try:
                    conn.execute("PRAGMA busy_timeout=2000")
                except Exception:
                    pass
                try:
                    conn.execute("PRAGMA foreign_keys=OFF")
                except Exception:
                    pass

                try:
                    tables = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                except Exception:
                    tables = []

                for (name,) in tables:
                    if not name:
                        continue
                    try:
                        conn.execute(f"DELETE FROM {name}")
                    except Exception:
                        # ignore per-table failures
                        pass

                try:
                    conn.commit()
                except Exception:
                    pass

                try:
                    conn.execute("VACUUM")
                except Exception:
                    pass
                return True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        # Always wipe metadata.db (robustly)
        metadata_cleared = False
        removed_dbs = 0
        try:
            if _try_delete_db_file(DB_PATH):
                removed_dbs += 1
                metadata_cleared = True
            else:
                # If Windows file locks prevent deletion, clear tables instead.
                try:
                    MetadataManager(DB_PATH)
                except Exception:
                    pass
                try:
                    conn = sqlite3.connect(DB_PATH)
                    try:
                        conn.execute("PRAGMA foreign_keys=OFF")
                        # Clear known tables
                        for tbl in ("image_tags", "tags", "favorites"):
                            try:
                                conn.execute(f"DELETE FROM {tbl}")
                            except Exception:
                                pass
                        try:
                            conn.commit()
                        except Exception:
                            pass
                        try:
                            conn.execute("VACUUM")
                        except Exception:
                            pass
                        metadata_cleared = True
                    finally:
                        conn.close()
                except Exception:
                    metadata_cleared = False
        except Exception:
            metadata_cleared = False

        removed_other_dbs = 0
        if wipe_other_dbs:
            try:
                for f in os.listdir(NODE_DIR):
                    if not f.lower().endswith(".db"):
                        continue
                    dbp = os.path.join(NODE_DIR, f)
                    if os.path.abspath(dbp) == os.path.abspath(DB_PATH):
                        continue
                    if _try_delete_db_file(dbp):
                        removed_other_dbs += 1
                    else:
                        # If locked, clear tables instead.
                        try:
                            if _clear_all_user_tables(dbp):
                                removed_other_dbs += 1
                        except Exception:
                            pass
            except Exception:
                pass

        try:
            metadata_manager = MetadataManager(DB_PATH)
        except Exception:
            pass

        try:
            PromptServer.instance.send_sync(
                "eros.cache.reset",
                {
                    "removed_files": removed_files,
                    "removed_dbs": removed_dbs,
                    "removed_other_dbs": removed_other_dbs,
                    "metadata_cleared": metadata_cleared,
                    "wipe_other_dbs": wipe_other_dbs,
                },
            )
        except Exception:
            pass

        return web.json_response(
            {
                "success": True,
                "removed_files": removed_files,
                "removed_dbs": removed_dbs,
                "removed_other_dbs": removed_other_dbs,
                "metadata_cleared": metadata_cleared,
                "wipe_other_dbs": wipe_other_dbs,
            }
        )
    except ValueError as ve:
        return web.json_response({"error": str(ve)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
