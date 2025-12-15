import os
import torch
import numpy as np
import json
from PIL import Image, ImageOps
import folder_paths
from server import PromptServer
from aiohttp import web
from .metadata_manager import MetadataManager

# Config & Persistence
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "eros_config.json")
DB_PATH = os.path.join(NODE_DIR, "metadata.db")

metadata_manager = MetadataManager(DB_PATH)

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


class CacheMapNode:
    @classmethod
    def INPUT_TYPES(s):
        default_path = os.path.join(folder_paths.get_input_directory(), "maps")
        input_config = {
            "required": {
                "cache_path": ("STRING", {"default": default_path, "tooltip": "Root directory for the cache. Maps will be stored/read from subfolders by type (e.g. cache_path/depth/filename.png)."}),
                "filename": ("STRING", {"forceInput": True, "tooltip": "The unique identifier (base filename) for the map. Use 'Load Image ErosDiffusion' to extract this from a source image."}),
                "map_type": (["auto"] + load_map_types() + ["browser"], {"default": "auto", "tooltip": "The type of map to handle. 'browser' is a pure pass-through."}),
                "save_if_new": ("BOOLEAN", {"default": True, "tooltip": "If True, saves the generated map to the cache directory if it wasn't found."}),
                "force_generation": ("BOOLEAN", {"default": False, "tooltip": "If True, ignores existing cache and forces regeneration + overwrite."}),
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

    def _get_cache_file_paths(self, cache_path, map_type, filename):
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
            # Scan all types
            for type_check in self._get_map_types():
                _, file_paths = self._get_cache_file_paths(cache_path, type_check, filename)
                if self._check_exists(file_paths):
                    # print(f"[CacheMap] Auto-Hit: Found {type_check} map for {filename}. Skipping generation.")
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
        
        # Browser Passthrough
        if map_type == "browser":
            img = kwargs.get("source_browser")
            if img is None:
                print("[CacheMap] Browser mode selected but no input connected/loaded.")
                return (torch.zeros((1, 512, 512, 3)),)
            return (img,)

        # Helper function to save tags
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
                
                # Notify frontend that tags were updated
                print(f"[CacheMap] Sending tag update notification to frontend for '{basename}'")
                PromptServer.instance.send_sync("eros.tags.updated", {
                    "basename": basename,
                    "tags": saved_tags
                })
            else:
                print(f"[CacheMap] No valid tags to process after filtering")


        # Generate All Logic
        if generate_all:
             print(f"[CacheMap] Processing 'Generate All' for {filename}...")
             for type_check in self._get_map_types():
                 if type_check == "custom":
                     continue
                 
                 source_img = kwargs.get(f"source_{type_check}")
                 if source_img is not None:
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
                         
                         # Save tags when generating/regenerating
                         save_tags_for_image(filename, tags_str)
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
                      
                      # Save tags for original image
                      save_tags_for_image(filename, tags_str)

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
                if kwargs.get(key) is not None:
                    generated_map = kwargs.get(key)
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

        return (generated_map,)

class CacheMapBrowserNode:
    @classmethod
    def INPUT_TYPES(s):
        default_path = os.path.join(folder_paths.get_input_directory(), "maps")
        return {
            "required": {
                "cache_path": ("STRING", {"default": default_path, "tooltip": "Root directory for the cache."}),
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
        
        # Check cache_path first
        p1 = os.path.join(cache_path, filename)
        if os.path.exists(p1):
            image_path = p1
        elif extra_path:
             p2 = os.path.join(extra_path, filename)
             if os.path.exists(p2):
                 image_path = p2
        
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
    if "path" not in request.rel_url.query:
        return web.json_response({"error": "Missing path parameter"}, status=400)
        
    target_path = request.rel_url.query["path"]
    if not os.path.exists(target_path):
         return web.json_response({"dirs": []})
    
    dirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
    return web.json_response({"dirs": sorted(dirs)})

@PromptServer.instance.routes.get("/eros/cache/fetch_files")
async def fetch_files(request):
    if "path" not in request.rel_url.query:
        return web.json_response({"error": "Missing path parameter"}, status=400)
    
    target_path = request.rel_url.query["path"]
    
    # Optional subfolder (e.g. map_type)
    subfolder = request.rel_url.query.get("subfolder", "")
    
    search_path = os.path.join(target_path, subfolder)
    
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
    if "path" not in request.rel_url.query or "filename" not in request.rel_url.query:
        return web.Response(status=400)
        
    target_path = request.rel_url.query["path"]
    filename = request.rel_url.query["filename"]
    subfolder = request.rel_url.query.get("subfolder", "")
    
    full_path = os.path.join(target_path, subfolder, filename)
    
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
