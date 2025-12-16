# ErosDiffusion — ComfyUI ControlNet Map Cache & Browser

**Handle your controlnet maps like a pro!**
Save resources by skipping already done work!
Generate, tag, cache, search and use ControlNet maps inside ComfyUI.


note: this is an **Alpha** stage plugin, if you want to support development feel free to [donate](https://donate.stripe.com/3cI7sDgZg4rr2Ln0HfcV202)

## Installation

1. **Clone** this folder into your ComfyUI `custom_nodes` directory:
   1. `cd <your comfy ui custom_nodes directory>`
   1. `git clone https://github.com/erosDiffusion/ErosDiffusion-ControlnetMaps.git`  
2. **Restart ComfyUI**: the nodes appear under the `ErosDiffusion` category or in the templates.
3. Open Settings->Eros and **switch to "Lit" interface**

**changelog 22.12.2025 - 17.22**

- There was a tiny initialization issue that would prevent you from opening the map browser if the file reference would be empty. this would be a common scenario for a new installation, and it is now solved.
please go to your extension directory, and in a terminal from the extension folder where you cloned do `git fetch` and `git pull` to get the newer version
- the good news is a fresh install of comfy requires no dependency install so it should work out of the box, the necessary sqlite db are created on the fly when missing. 




![mapbrowser0](https://github.com/user-attachments/assets/55a1a07b-c0ae-45f1-bc40-0f2d03760ffb)

### Video preview for usage
https://github.com/user-attachments/assets/f7881a43-2fc4-41ca-9655-8090d4b42c64

## Overview

This package provides two ComfyUI nodes that simplify working with ControlNet maps by caching generated maps to disk and exposing a visual browser as well as tag them and store references in a local sqlite db:

- `CacheMapNode` — smart cache layer that checks a disk cache before requesting expensive preprocessors and can save generated maps for reuse.
- `CacheMapBrowserNode` — sidebar/browser integration to preview and load cached maps (returns the selected map).

 **note**: there are 2 frontend implementations, please use the LIT one , switch in settings:
 <img width="1087" height="1028" alt="image" src="https://github.com/user-attachments/assets/5b0024ad-8f54-4bdd-b2e9-7b28e82fb36b" />

These nodes are designed to reduce repeated preprocessing work and help organize map assets by filename and type.
Ideally later share and reuse high quality maps in a standard format.

## Key Features

- Cache lookup by filename and map type (supports multiple extensions).
- `auto` mode detects existing map types and only runs connected preprocessors when needed (returns the first connected, top to bottom)
- `generate_all` option to batch-save all connected preprocessors and the original image tags them and saves all maps to cache folder
- Tagging: comma-separated `tags` input is persisted to a lightweight metadata DB for later retrieval and UI updates. you can connect an llm to the source image and have comma separated list of tags of your choice
- Browse and search the nodes by tag or type and easily select the image for reuse.

## Nodes

**CacheMapNode**

- Category: `ErosDiffusion`
- Returns: `IMAGE` (`map`)
- Useful inputs (see node UI for full list): `cache_path`, `filename`, `map_type` (auto | depth | canny | ... | browser), `save_if_new`, `force_generation`, `generate_all`, `tags`, `source_<map_type>` image inputs, `source_original`.
- Behavior summary:
  - If `map_type` is `browser` the node simply passes through the `source_browser` image.
  - In `auto` mode it searches cache subfolders for existing maps and only requests the necessary preprocessor inputs on cache miss.
  - `save_if_new` saves generated maps to `<cache_path>/<type>/<basename>.png`.
  - `generate_all` forces evaluation of all connected preprocessor inputs and saves them.
  - Tags are parsed, deduplicated, and stored via the included metadata manager.

**CacheMapBrowserNode**

- Category: `ErosDiffusion`
- Returns: `IMAGE, MASK` (`image, mask`)
- Opens a map browser in the ComfyUI sidebar (button on the node). Select a file and the node loads the image and corresponding mask (if any).

## Default Map Types

Default map types are configured in `eros_config.json`. The shipped defaults are:

```
depth, canny, openpose, lineart, scribble, softedge, normal, seg, shuffle, mediapipe_face, custom
```

You can customize that list in `eros_config.json` (node will read this file on load). (this feature should work but not well tested)
custom is a passthrough, you can pass whatever you want. original is always meatn to be there, as you need it for overlays. 



## Typical Workflows (two workflows preset in the workflow examples folder)

- Quick reuse: Connect a preprocessor node to `CacheMapNode`'s `source_<type>` and set `filename` to a stable identifier; future runs will load the cached map instead of re-running the preprocessor.
- Batch export: Enable `generate_all` to save all connected preprocessor outputs and the original image to disk.
- Browse & inject: Use the `CacheMapBrowserNode` to visually pick maps and feed them into downstream nodes.

  <img width="1424" height="1125" alt="image" src="https://github.com/user-attachments/assets/774d7696-80ee-4136-b094-a540d82a7cab" />


## Remarks

- ctrl+click on existing tag adds to the current selected
- tags are not duplicated
- comma separate tags in input and hit enter to insert or provide at generation time
- filter using tags
- type selection works but lable does not highlight
- there are two implementations, vanilla and lit, lit is the target you can switch in the preferences (but vanilla might be broken now)

## Paths & Defaults

- Default cache root (node UI): `<ComfyUI input directory>/maps` (can be changed in the node's `cache_path`).
- Cache layout: `<cache_path>/<map_type>/<basename>.png` and originals under `<cache_path>/original/`.

## Tags & Metadata

Provide a comma-separated `tags` string to `CacheMapNode` and tags will be parsed, normalized (deduplicated case-insensitive), and stored in a local metadata DB (see `metadata_manager.py`). The node also notifies the frontend when tags are updated.

## Troubleshooting

- If cached maps are not being found, verify `cache_path` and that filenames use the same basename (node strips extensions when checking).
- For `auto` mode misses, ensure the corresponding `source_<type>` input is connected so the preprocessor can run and generate the map.
- If browser doesn't show files, confirm the UI has access to the configured `cache_path` and any `extra_path` you provided.

## Files of interest

- `cache_map_nodes.py` — node implementations (CacheMapNode, CacheMapBrowserNode)
- `eros_config.json` — map type configuration
- `metadata_manager.py` — tagging/DB helper
- `requirements.md` — dependency notes

## Contributing

PRs welcome. Keep changes focused and add tests/examples where possible. Test it and tell me what's wrong.
during the vanilla to lit porting something got off, so there are some regressions.







