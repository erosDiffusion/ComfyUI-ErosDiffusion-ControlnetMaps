/**
 * Author: ErosDiffusion (EF)
 * Email: erosdiffusionai+controlnetmaps@gmail.com
 * Year: 2025
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { DRAWER_CSS } from "./cache_map_styles.js";
import { CacheService } from "./cache_service.js";

// ========================================================
// 1. Web Components Implementation
// ========================================================
// CacheService is now imported

// ========================================================
// 2. Components
// ========================================================

class ErosBrowserControls extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
  }

  connectedCallback() {
    this.render();
    this.setupEvents();
  }

  render() {
    // Shared styles needed? Or just minimal styles here.
    // We'll reuse DRAWER_CSS via adoption if supported or just link it.
    // For simplicity, we assume parent injects styles or we add them.
    this.shadowRoot.innerHTML = `
            <style>${DRAWER_CSS}</style>
            <div class="eros-controls">
                <div class="eros-tabs" id="tabs-container"></div>
                <div class="eros-control-grid">
                    <div style="display:flex; flex-direction:column; gap:4px;">
                        <label class="eros-overlay-toggle"><input type="checkbox" id="chk-overlay"> Overlay Original</label>
                        <label class="eros-overlay-toggle"><input type="checkbox" id="chk-badges" checked> Show Badges</label>
                        <label class="eros-overlay-toggle"><input type="checkbox" id="chk-cache" checked> Cache Busting</label>
                    </div>
                    <label class="eros-overlay-controls">Blend Mode <select id="sel-blend"></select></label>
                    <label class="eros-overlay-controls">Opacity <input type="range" id="rng-opacity" min="0" max="1" step="0.01"></label>
                    <label class="eros-overlay-controls">Columns <input type="range" id="rng-columns" min="1" max="8" step="1" value="4"></label>
                    <label class="eros-overlay-controls">Badge Size <input type="range" id="rng-badgesize" min="6" max="16" step="1" value="9"></label>
                    <button class="eros-btn" id="btn-refresh" style="height:100%;">↻ Refresh</button>
                </div>
            </div>`;

    this.renderTabs();
    this.renderBlendModes();
  }

  renderTabs() {
    const tabs = [
      "depth",
      "canny",
      "lineart",
      "openpose",
      "scribble",
      "softedge",
      "normal",
      "segmentation",
      "original",
    ];
    const container = this.shadowRoot.getElementById("tabs-container");
    container.innerHTML = "";
    tabs.forEach((tab) => {
      const el = document.createElement("div");
      el.className = `eros-tab`;
      el.innerText = tab;
      el.onclick = () => {
        this.shadowRoot
          .querySelectorAll(".eros-tab")
          .forEach((t) => t.classList.remove("active"));
        el.classList.add("active");
        this.dispatchEvent(new CustomEvent("tab-changed", { detail: { tab } }));
      };
      container.appendChild(el);
    });
    // Default active? Controlled by parent calling methods ideally, but internal state fine for UI.
    container.firstChild?.classList.add("active");
  }

  renderBlendModes(modesStr) {
    const modes = (
      modesStr || "normal,multiply,screen,overlay,luminosity,difference"
    ).split(",");
    const sel = this.shadowRoot.getElementById("sel-blend");
    if (!sel) return;
    sel.innerHTML = ""; // clear previous
    modes.forEach((m) => sel.add(new Option(m.trim(), m.trim())));
    sel.value = "luminosity";
  }

  set config(cfg) {
    // Update controls based on config if needed
    const root = this.shadowRoot;
    if (!root) return;

    if (cfg.blendModes) this.renderBlendModes(cfg.blendModes);

    // Update inputs to match state
    const setVal = (id, val) => {
      const el = root.getElementById(id);
      if (el) el.value = val;
    };
    const setChk = (id, val) => {
      const el = root.getElementById(id);
      if (el) el.checked = !!val;
    };

    setChk("chk-overlay", cfg.overlayEnabled);
    setChk("chk-badges", cfg.showTagBadges);
    setChk("chk-cache", cfg.cacheBusting);
    setVal("sel-blend", cfg.blendMode);
    setVal("rng-opacity", cfg.opacity);
    setVal("rng-columns", cfg.columns);
    setVal("rng-badgesize", cfg.badgeSize);

    // Restore active tab visual state
    if (cfg.currentTab) {
      root.querySelectorAll(".eros-tab").forEach((t) => {
        if (t.innerText === cfg.currentTab) t.classList.add("active");
        else t.classList.remove("active");
      });
    }
  }

  setupEvents() {
    const root = this.shadowRoot;
    const emit = (name, val) =>
      this.dispatchEvent(new CustomEvent(name, { detail: val }));

    const emitDebounced = (name, val) => {
      // Simple debouncing for sliders
      const key = name + (val.key || "");
      if (this._timers[key]) clearTimeout(this._timers[key]);
      this._timers[key] = setTimeout(() => emit(name, val), 30); // 30ms debounce
    };
    this._timers = {};

    root.getElementById("chk-overlay").onchange = (e) =>
      emit("setting-change", {
        key: "overlayEnabled",
        value: e.target.checked,
      });
    root.getElementById("chk-badges").onchange = (e) =>
      emit("setting-change", { key: "showTagBadges", value: e.target.checked });
    root.getElementById("chk-cache").onchange = (e) =>
      emit("setting-change", { key: "cacheBusting", value: e.target.checked });
    root.getElementById("sel-blend").onchange = (e) =>
      emit("setting-change", { key: "blendMode", value: e.target.value });
    root.getElementById("rng-opacity").oninput = (e) =>
      emitDebounced("setting-change", {
        key: "opacity",
        value: parseFloat(e.target.value),
      });
    root.getElementById("rng-columns").oninput = (e) =>
      emitDebounced("setting-change", {
        key: "columns",
        value: parseInt(e.target.value),
      });
    root.getElementById("rng-badgesize").oninput = (e) =>
      emitDebounced("setting-change", {
        key: "badgeSize",
        value: parseInt(e.target.value),
      });
    root.getElementById("btn-refresh").onclick = () =>
      emit("refresh-requested");
  }
}
customElements.define("eros-browser-controls", ErosBrowserControls);

class ErosImageGrid extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
  }

  connectedCallback() {
    this.shadowRoot.innerHTML = `<style>${DRAWER_CSS}</style><div class="eros-grid" id="grid"></div>`;
    this.grid = this.shadowRoot.getElementById("grid");
  }

  set config(cfg) {
    if (!this.grid) return;
    if (cfg.columns) this.grid.style.setProperty("--grid-columns", cfg.columns);
    if (cfg.badgeSize)
      this.grid.style.setProperty("--badge-font-size", cfg.badgeSize + "px");
    this.cfg = cfg; // store for rendering
    if (this.lastFiles)
      this.render(
        this.lastFiles,
        this.lastImgTags,
        this.lastCachePath,
        this.lastTab,
        this.lastSelected
      );
  }

  render(files, imageTags, cachePath, currentTab) {
    this.lastFiles = files;
    this.lastImgTags = imageTags;
    this.lastCachePath = cachePath;
    this.lastTab = currentTab;
    if (!this.grid) return;

    // Improve performace: fragment
    const frag = document.createDocumentFragment();
    files.forEach((f) => {
      const item = document.createElement("div");
      item.className = "eros-item";

      // Cache Busting Logic
      const ts = this.cfg?.cacheBusting ? `&t=${Date.now()}` : "";
      const imgPath = `/eros/cache/view_image?path=${encodeURIComponent(
        cachePath
      )}&subfolder=${currentTab}&filename=${encodeURIComponent(f)}${ts}`;

      // Layout: Loader + Image
      // We use opacity transition for smooth load
      let html = `
                <div class="eros-loader"></div>
                <img src="${imgPath}" loading="lazy" style="opacity:0; transition:opacity 0.2s;" 
                     onload="this.style.opacity='1'; this.previousElementSibling.style.display='none';">
            `;

      if (this.cfg?.overlayEnabled) {
        const ovPath = `/eros/cache/view_image?path=${encodeURIComponent(
          cachePath
        )}&subfolder=original&filename=${encodeURIComponent(f)}${ts}`;
        html += `<img class="eros-overlay" src="${ovPath}" 
                        style="position:absolute; top:0; left:0; width:100%; height:100%; opacity:${this.cfg.opacity}; mix-blend-mode:${this.cfg.blendMode}; pointer-events:none;"
                        onerror="this.style.display='none'">`;
      }

      const tags = imageTags.get(
        f
          .split("/")
          .pop()
          .replace(/\.[^/.]+$/, "")
      );
      if (this.cfg?.showTagBadges && tags && tags.size > 0) {
        html += `<div class="eros-tag-badges">`;
        Array.from(tags)
          .slice(0, 6)
          .forEach((t) => (html += `<span class="eros-tag-badge">${t}</span>`));
        if (tags.size > 6)
          html += `<span class="eros-tag-badge" style="background:#555;">+${
            tags.size - 6
          }</span>`;
        html += `</div>`;
      }
      item.innerHTML =
        html +
        `<div class="eros-item-label">${f
          .split("/")
          .pop()
          .replace(/\.[^/.]+$/, "")}</div>`;

      item.onclick = () => {
        this.shadowRoot
          .querySelectorAll(".eros-item")
          .forEach((i) => i.classList.remove("selected"));
        item.classList.add("selected");
        this.dispatchEvent(
          new CustomEvent("image-selected", {
            detail: { filename: f, imgPath },
          })
        );
      };
      frag.appendChild(item);
    });
    this.grid.innerHTML = "";
    this.grid.appendChild(frag);
  }
}
customElements.define("eros-image-grid", ErosImageGrid);

class ErosTagSidebar extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.collapsed = { filter: false, selected: false };
  }

  connectedCallback() {
    this.shadowRoot.innerHTML = `<style>${DRAWER_CSS}</style>
            <div class="eros-tag-sidebar">
                <div class="eros-tag-section">
                    <div class="eros-tag-section-label" id="btn-toggle-filter">Filter by Tags <span id="icon-filter">▼</span></div>
                    <div class="eros-tag-section-content" id="filter-content">
                        <input class="eros-tag-input" id="inp-tag-search" placeholder="Search tags..." style="margin-bottom:4px;">
                        <div class="eros-tag-chips" id="filter-chips"></div>
                    </div>
                </div>
                <div class="eros-tag-section">
                    <div class="eros-tag-section-label" id="btn-toggle-selected">Selected Tags <span id="icon-selected">▼</span></div>
                    <div class="eros-tag-section-content" id="selected-content">
                        <div class="eros-tag-chips" id="selected-tags-chips"></div>
                        <div class="eros-tag-input-row" id="tag-input-row" style="display:none;">
                            <input class="eros-tag-input" id="inp-new-tag" placeholder="Add tag...">
                            <button class="eros-tag-add-btn" id="btn-add-tag">+</button>
                            <button class="eros-tag-add-btn" id="btn-auto-tag">🤖</button>
                        </div>
                    </div>
                </div>
            </div>`;
    this.setupEvents();
  }

  setupEvents() {
    const root = this.shadowRoot;
    // Toggles
    root.getElementById("btn-toggle-filter").onclick = () => {
      this.collapsed.filter = !this.collapsed.filter;
      root.getElementById("filter-content").style.display = this.collapsed
        .filter
        ? "none"
        : "flex";
      root.getElementById("icon-filter").innerText = this.collapsed.filter
        ? "▶"
        : "▼";
    };
    root.getElementById("btn-toggle-selected").onclick = () => {
      this.collapsed.selected = !this.collapsed.selected;
      root.getElementById("selected-content").style.display = this.collapsed
        .selected
        ? "none"
        : "flex";
      root.getElementById("icon-selected").innerText = this.collapsed.selected
        ? "▶"
        : "▼";
    };

    // Search
    root.getElementById("inp-tag-search").oninput = (e) => {
      const val = e.target.value;
      clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => {
        this.dispatchEvent(new CustomEvent("filter-search", { detail: val }));
      }, 300);
    };

    // Add Tag
    const addTag = () => {
      const val = root.getElementById("inp-new-tag").value.trim();
      if (val) {
        this.dispatchEvent(new CustomEvent("tag-add", { detail: val }));
        root.getElementById("inp-new-tag").value = "";
      }
    };
    root.getElementById("inp-new-tag").onkeydown = (e) => {
      if (e.key === "Enter") addTag();
    };
    root.getElementById("btn-add-tag").onclick = addTag;
    root.getElementById("btn-auto-tag").onclick = () =>
      this.dispatchEvent(new CustomEvent("tag-auto"));
  }

  renderFilters(allTags, activeFilters, query) {
    const container = this.shadowRoot.getElementById("filter-chips");
    if (!container) return;
    container.innerHTML = "";

    // Filter logic is mostly UI specific here (display) but could be pre-filtered
    allTags.forEach((count, name) => {
      if (!name || (query && !name.toLowerCase().includes(query.toLowerCase())))
        return;
      const chip = document.createElement("div");
      chip.className = `eros-tag-chip ${
        activeFilters.has(name) ? "active" : ""
      }`;
      chip.innerHTML = `${name} <span style="opacity:0.6;font-size:9px;">${count}</span>`;
      chip.onclick = (e) =>
        this.dispatchEvent(
          new CustomEvent("filter-click", {
            detail: { name, ctrlKey: e.ctrlKey },
          })
        );
      container.appendChild(chip);
    });
  }

  renderSelected(tags, hasSelection) {
    const chips = this.shadowRoot.getElementById("selected-tags-chips");
    const row = this.shadowRoot.getElementById("tag-input-row");
    if (!chips || !row) return; // check both

    if (!hasSelection) {
      chips.innerHTML =
        '<div style="font-size:10px; color:#666;">Select an image to manage tags</div>';
      row.style.display = "none";
      return;
    }
    row.style.display = "flex";
    chips.innerHTML = "";

    if (!tags || tags.size === 0) {
      chips.innerHTML =
        '<div style="font-size:10px; color:#888; font-style:italic;">No tags</div>';
    } else {
      tags.forEach((t) => {
        const chip = document.createElement("div");
        chip.className = "eros-tag-chip";
        chip.innerHTML = `${t} <span class="eros-tag-chip-remove">×</span>`;
        chip.querySelector(".eros-tag-chip-remove").onclick = (e) => {
          e.stopPropagation();
          this.dispatchEvent(new CustomEvent("tag-remove", { detail: t }));
        };
        chips.appendChild(chip);
      });
    }
  }
}
customElements.define("eros-tag-sidebar", ErosTagSidebar);

class ErosCacheBrowser extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.cache = new CacheService();

    // Default Settings
    const defaults = {
      overlayEnabled: false,
      showTagBadges: true,
      blendMode: "luminosity",
      opacity: 0.25,
      columns: 4,
      badgeSize: 9,
      cacheBusting: true,
      blendModes:
        "normal,multiply,screen,overlay,darken,lighten,color-dodge,color-burn,hard-light,soft-light,difference,exclusion,hue,saturation,color,luminosity",
      currentTab: "depth", // Persist active tab too
    };

    // Load from LocalStorage
    let stored = {};
    try {
      stored = JSON.parse(
        localStorage.getItem("eros_cache_browser_settings") || "{}"
      );
    } catch (e) {
      console.warn("Failed to load settings", e);
    }

    this.settings = { ...defaults, ...stored };

    this.state = {
      activeFilters: new Set(),
      tagSearchQuery: "",
      currentFilename: null,
      isOpen: false,
      currentTab: this.settings.currentTab, // Init tab from settings
      files: [],
    };
  }

  saveSettings() {
    try {
      localStorage.setItem(
        "eros_cache_browser_settings",
        JSON.stringify(this.settings)
      );
    } catch (e) {
      console.warn("Failed to save settings", e);
    }
  }

  connectedCallback() {
    console.log("[WC] ErosCacheBrowser connectedCallback");
    this.render();
    this.setupOrchestration();
    // Global listener for escape?
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.state.isOpen) this.close();
    });
  }

  render() {
    this.shadowRoot.innerHTML = `
            <style>${DRAWER_CSS}
            .eros-wc-container { display: flex; height: 100%; overflow: hidden; width: 100%; }
            .eros-main-column { display: flex; flex-direction: column; flex: 1; overflow: hidden; position: relative; }
            eros-browser-controls { flex-shrink: 0; }
            eros-image-grid { flex: 1; overflow: hidden; position: relative; display: block; }
            eros-tag-sidebar { width: 300px; display: block; border-left: 1px solid #333; background: rgba(0,0,0,0.2); flex-shrink: 0; }
            </style>
            <div class="eros-drawer" id="drawer" style="display:none;">
                 <div class="eros-drawer-resize-handle" id="resize-handle"></div>
                 
                 <div class="eros-drawer-header">
                    <h3>Cache Browser</h3>
                    <div class="eros-drawer-close" id="btn-close">×</div>
                 </div>

                 <div class="eros-wc-container">
                    <div class="eros-main-column">
                        <eros-browser-controls id="controls"></eros-browser-controls>
                        <eros-image-grid id="grid-comp"></eros-image-grid>
                    </div>
                    <eros-tag-sidebar id="sidebar-comp"></eros-tag-sidebar>
                 </div>
            </div>
            <div class="eros-modal-bg" id="modal-bg" style="display:none;"></div>
        `;

    // Element Refs
    this.els = {
      drawer: this.shadowRoot.getElementById("drawer"),
      bg: this.shadowRoot.getElementById("modal-bg"),
      controls: this.shadowRoot.getElementById("controls"),
      grid: this.shadowRoot.getElementById("grid-comp"),
      sidebar: this.shadowRoot.getElementById("sidebar-comp"),
      resize: this.shadowRoot.getElementById("resize-handle"),
      btnClose: this.shadowRoot.getElementById("btn-close"),
    };

    // Init Resize
    this.initResize();
  }

  setupOrchestration() {
    const { controls, grid, sidebar, bg, resize } = this.els;

    // 1. Init Controls with settings
    controls.config = this.settings;
    grid.config = this.settings; // Ensure grid gets initial settings

    // 2. Controls -> Settings
    controls.addEventListener("setting-change", (e) => {
      const { key, value } = e.detail;
      this.settings[key] = value;
      this.saveSettings(); // Persist
      grid.config = this.settings;
      // If we update blend modes dynamically we might need to push back to controls
      // but for now it is static config
    });
    controls.addEventListener("refresh-requested", () => this.fetchFiles());
    controls.addEventListener("tab-changed", (e) => {
      this.state.currentTab = e.detail.tab;
      this.settings.currentTab = e.detail.tab; // Sync setting
      this.saveSettings(); // Persist
      this.fetchFiles();
    });

    // 2. Sidebar -> Filter/Tags
    sidebar.addEventListener("filter-search", (e) => {
      this.state.tagSearchQuery = e.detail;
      this.renderGrid();
    });
    sidebar.addEventListener("tag-add", (e) =>
      this.cache.addTag(this.getBasename(this.state.currentFilename), e.detail)
    );
    sidebar.addEventListener("tag-remove", (e) =>
      this.cache.removeTag(
        this.getBasename(this.state.currentFilename),
        e.detail
      )
    );
    sidebar.addEventListener("tag-auto", () =>
      this.cache.autoTag(this.getBasename(this.state.currentFilename))
    );

    sidebar.addEventListener("filter-click", (e) => {
      const { name, ctrlKey } = e.detail;
      if (ctrlKey) {
        this.cache.addTag(this.getBasename(this.state.currentFilename), name);
      } else {
        if (this.state.activeFilters.has(name))
          this.state.activeFilters.delete(name);
        else this.state.activeFilters.add(name);
        this.renderGrid();
      }
    });

    // 3. Grid -> selection
    grid.addEventListener("image-selected", (e) => {
      this.state.currentFilename = e.detail.filename;
      this.updateSidebarSelection();

      // Node Update Logic
      if (this.activeNode) {
        // Fixed: removed .widgets check, safer below
        const w = this.activeNode.widgets?.find((w) => w.name === "filename");
        if (w) {
          w.value = `${this.state.currentTab}/${e.detail.filename}`;
          w.callback?.(w.value);
          const img = new Image();
          img.onload = () => {
            this.activeNode.imgs = [img];
            app.graph.setDirtyCanvas(true);
          };
          img.src = e.detail.imgPath;
        }
      }
    });

    // 4. Service -> UI
    this.cache.subscribe((evt, data) => {
      if (evt === "tags-loaded") {
        this.updateSidebarFilters();
      }
      if (evt === "tag-added" || evt === "tag-removed") {
        this.updateSidebarFilters();
        this.updateSidebarSelection();
        if (this.state.files.length) this.pushGridData();
      }
    });

    bg.onclick = () => this.close();
    if (this.els.btnClose) this.els.btnClose.onclick = () => this.close();
  }

  initResize() {
    // resize handle is .eros-drawer-resize-handle inside shadowRoot
    const resizeHandle = this.shadowRoot.getElementById("resize-handle");
    const drawer = this.els.drawer;
    if (!resizeHandle || !drawer) return;

    let isResizing = false;

    resizeHandle.onmousedown = (e) => {
      isResizing = true;
      e.preventDefault();
      document.body.style.cursor = "ew-resize";
    };

    window.addEventListener("mousemove", (e) => {
      if (!isResizing) return;
      // Calculations for right-aligned drawer
      let w = window.innerWidth - e.clientX;
      // Constraints
      if (w < 300) w = 300;
      if (w > window.innerWidth - 50) w = window.innerWidth - 50;
      drawer.style.width = w + "px";
    });

    window.addEventListener("mouseup", () => {
      if (isResizing) {
        isResizing = false;
        document.body.style.cursor = "";
      }
    });
  }

  open(node) {
    this.activeNode = node;
    this.cache.setCachePath(
      node?.widgets?.find((w) => w.name === "cache_path")?.value || ""
    );

    // Initial Selection
    if (node) {
      const wVal = node.widgets?.find((w) => w.name === "filename")?.value;
      if (wVal) {
        const parts = wVal.split("/");
        if (parts.length > 1) {
          // Try to sync tab
          const tab = parts[0];
          // Check against known tabs or just assume valid subfolder?
          // Let's check config tabs later maybe, for now simple heuristics
          this.state.currentTab = tab;
          this.state.currentFilename = parts[1];
        } else {
          this.state.currentFilename = wVal;
        }
      }
    }

    this.state.isOpen = true;
    this.els.drawer.style.display = "block";
    setTimeout(() => this.els.drawer.classList.add("open"), 10);
    this.els.bg.style.display = "block";
    this.fetchFiles();
    this.cache.loadTags();
  }

  close() {
    this.els.drawer.style.transform = "translateX(100%)";
    setTimeout(() => {
      if (!this.state.isOpen) this.els.drawer.style.display = "none";
    }, 300); // match transition
    this.els.bg.style.display = "none";
    this.state.isOpen = false;
    this.activeNode = null;
  }

  getBasename(filename) {
    return filename
      ? filename
          .split("/")
          .pop()
          .replace(/\.[^/.]+$/, "")
      : "";
  }

  async fetchFiles() {
    if (!this.state.isOpen) return;
    this.cache.fetchFiles(this.state.currentTab).then(async (files) => {
      this.state.files = files;
      this.renderGrid();

      // Fetch tags for these files
      // Ideally backend would return them, but for now we parallel fetch
      // or we use a bulk endpoint if available.
      // Previous code used lazy loading or a bulk load?
      // "loadImageTags" fetches for ONE image. We need it for ALL.
      // Or we iterate.

      // Let's iterate but be gentle? or assume we need to load them to show badges.
      // Better: loop and load.
      for (const f of files) {
        const base = this.getBasename(f);
        if (!this.cache.imageTags.has(base)) {
          // Fire and forget, or await?
          // Fire and forget will trigger 'tag-added'? No, loadImageTags returns set.
          // We need to manually notify or update.
          await this.cache.loadImageTags(base);
        }
      }
      // Force re-render after loading tags?
      // loadImageTags updates the map. We need to trigger a render.
      this.renderGrid();
      this.updateSidebarSelection(); // Update sidebar if something selected
    });
  }

  renderGrid() {
    let files = this.state.files;
    const query = (this.state.tagSearchQuery || "").toLowerCase();

    // Filter
    if (query || this.state.activeFilters.size > 0) {
      files = files.filter((f) => {
        const base = this.getBasename(f);
        const tags = this.cache.imageTags.get(base);

        // Search Query
        if (query) {
          let match = base.toLowerCase().includes(query);
          if (!match && tags) {
            for (let t of tags)
              if (t.toLowerCase().includes(query)) {
                match = true;
                break;
              }
          }
          if (!match) return false;
        }

        // Active Filters
        if (this.state.activeFilters.size > 0) {
          if (!tags) return false;
          for (let ft of this.state.activeFilters)
            if (!tags.has(ft)) return false;
        }
        return true;
      });
    }

    // Pass to Grid
    this.els.grid.render(
      files,
      this.cache.imageTags,
      this.cache.cachePath,
      this.state.currentTab,
      this.state.currentFilename
    );
    this.updateSidebarFilters();
  }

  // Helpers to bridge Grid/State -> Sidebar
  pushGridData() {
    this.els.grid.render(
      this.state.files,
      this.cache.imageTags,
      this.cache.cachePath,
      this.state.currentTab,
      this.state.currentFilename
    );
    this.renderGrid();
  }

  updateSidebarFilters() {
    this.els.sidebar.renderFilters(
      this.cache.allTags,
      this.state.activeFilters,
      this.state.tagSearchQuery
    );
  }

  updateSidebarSelection() {
    const base = this.getBasename(this.state.currentFilename);
    const tags = this.cache.imageTags.get(base);
    this.els.sidebar.renderSelected(
      tags || new Set(),
      !!this.state.currentFilename
    );
  }
}
customElements.define("eros-cache-browser", ErosCacheBrowser);
