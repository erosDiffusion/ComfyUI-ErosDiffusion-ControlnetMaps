/**
 * Author: ErosDiffusion (EF)
 * Email: erosdiffusionai+controlnetmaps@gmail.com
 * Year: 2025
 */

import { app } from "../../scripts/app.js";
import {
  html,
  css,
  LitElement,
} from "https://cdn.jsdelivr.net/gh/lit/dist@3/core/lit-core.min.js";
import { DRAWER_CSS } from "./cache_map_styles.js";
import { CacheService } from "./cache_service.js";

// Share styles via Constructable Stylesheets if possible, or inject
// Lit supports 'styles' static property.
// We reuse DRAWER_CSS but need to wrap it in css tagged template?
// DRAWER_CSS is a string. We can use unsafeCSS if we trust it, or just inject logic.

// Small button style injection
const BTN_CSS = `
.eros-btn-small {
    background: #444; color: white; border: none; border-radius: 4px; cursor: pointer;
    transition: background 0.2s;
}
.eros-btn-small:hover { background: #666; }
`;

// Simpler: Use shadow DOM styles.

const sharedStyles = css`
  :host {
    display: block;
    box-sizing: border-box;
  }
  /* We inject the global drawer CSS here for components that need it */
`;

// Centralized list of map types / tabs (should match node's map types)
const MAP_TABS = [
  "original",
  "depth",
  "canny",
  "openpose",
  "lineart",
  "scribble",
  "softedge",
  "normal",
  "seg",
  "shuffle",
  "mediapipe_face",
  "custom",
];

// ========================================================
// Lit Browser Controls
// ========================================================
class ErosLitControls extends LitElement {
  static properties = {
    config: { type: Object }, // Input config
  };

  constructor() {
    super();
    this.config = {};
    this._debouncers = {};
  }

  // Helper to emit events
  emit(name, detail) {
    this.dispatchEvent(
      new CustomEvent(name, { detail, bubbles: true, composed: true })
    );
  }

  emitDebounced(name, detail, key) {
    if (this._debouncers[key]) clearTimeout(this._debouncers[key]);
    this._debouncers[key] = setTimeout(() => this.emit(name, detail), 30);
  }

  render() {
    const c = this.config || {};
    // Note: In Lit, we bind values.
    return html`
      <style>
        ${DRAWER_CSS} /* Inject raw CSS string */
                :host {
          display: block;
        }
      </style>
      <div class="eros-controls">
        <div class="eros-tabs">
          ${MAP_TABS.map(
            (t) => html`
              <div
                class="eros-tab ${t === c.currentTab ? "active" : ""}"
                @click=${() => this.emit("tab-changed", { tab: t })}
              >
                ${t}
              </div>
            `
          )}
        </div>

        <div class="eros-control-grid">
          <div style="display:flex; flex-direction:column; gap:4px;">
            <label class="eros-overlay-toggle">
              <input
                type="checkbox"
                .checked=${!!c.overlayEnabled}
                @change=${(e) =>
                  this.emit("setting-change", {
                    key: "overlayEnabled",
                    value: e.target.checked,
                  })}
              />
              Overlay Original
            </label>
            <label class="eros-overlay-toggle">
              <input
                type="checkbox"
                .checked=${!!c.showTagBadges}
                @change=${(e) =>
                  this.emit("setting-change", {
                    key: "showTagBadges",
                    value: e.target.checked,
                  })}
              />
              Show Badges
            </label>
            <label class="eros-overlay-toggle">
              <input
                type="checkbox"
                .checked=${!!c.cacheBusting}
                @change=${(e) =>
                  this.emit("setting-change", {
                    key: "cacheBusting",
                    value: e.target.checked,
                  })}
              />
              Cache Busting
            </label>
          </div>

          <label class="eros-overlay-controls">
            Blend Mode
            <select
              .value=${c.blendMode || "luminosity"}
              @change=${(e) =>
                this.emit("setting-change", {
                  key: "blendMode",
                  value: e.target.value,
                })}
            >
              ${(c.blendModes || "normal,luminosity")
                .split(",")
                .map(
                  (m) => html`<option value=${m.trim()}>${m.trim()}</option>`
                )}
            </select>
          </label>

          <label class="eros-overlay-controls">
            Opacity
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              .value=${c.opacity || 0.25}
              @input=${(e) =>
                this.emitDebounced(
                  "setting-change",
                  { key: "opacity", value: parseFloat(e.target.value) },
                  "opacity"
                )}
            />
          </label>

          <label class="eros-overlay-controls">
            Columns
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              .value=${c.columns || 4}
              @input=${(e) =>
                this.emitDebounced(
                  "setting-change",
                  { key: "columns", value: parseInt(e.target.value) },
                  "columns"
                )}
            />
          </label>

          <label class="eros-overlay-controls">
            Badge Size
            <input
              type="range"
              min="8"
              max="16"
              step="1"
              .value=${c.badgeSize || 9}
              @input=${(e) =>
                this.emitDebounced(
                  "setting-change",
                  { key: "badgeSize", value: parseInt(e.target.value) },
                  "badgeSize"
                )}
            />
          </label>

          <button
            class="eros-btn"
            @click=${() => this.emit("refresh-requested")}
          >
            Refresh
          </button>
        </div>
      </div>
    `;
  }
}
customElements.define("eros-lit-controls", ErosLitControls);

// ========================================================
// Lit Image Grid
// ========================================================
class ErosLitGrid extends LitElement {
  static properties = {
    files: { type: Array },
    imageTags: { type: Object }, // Map
    cachePath: { type: String },
    currentTab: { type: String },
    config: { type: Object },
    selectedFilename: { type: String },
  };

  constructor() {
    super();
    this.files = [];
    this.config = {};
  }

  render() {
    const c = this.config || {};
    const cols = c.columns || 4;
    const badgeSize = c.badgeSize || 9;

    return html`
      <style>
        ${DRAWER_CSS} :host {
          display: block;
          flex: 1;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }
        .eros-grid {
          grid-template-columns: repeat(${cols}, 1fr);

          --badge-font-size: ${badgeSize}px;

          scrollbar-width: none; /* Firefox */
          -ms-overflow-style: none; /* IE 10+ */
        }
        .eros-grid::-webkit-scrollbar {
          width: 0;
          height: 0;
          display: none;
        }
      </style>
      <div class="eros-grid">
        ${this.files.map((f) => this.renderItem(f, c))}
      </div>
    `;
  }

  renderItem(f, c) {
    // Support filenames that may be prefixed with a subfolder: 'type/file'
    let sub = this.currentTab;
    let filename = f;
    if (f && f.includes("/")) {
      const parts = f.split("/");
      sub = parts[0];
      filename = parts.slice(1).join("/");
    }
    const base = filename.replace(/\.[^/.]+$/, "");
    const ts = c.cacheBusting ? `&t=${Date.now()}` : "";
    // Note: Using Date.now() in render() forces reload on every update?
    // Better to store timestamp in state if we want to control it, implies simple cache busting might cause reloads.
    // For now adhering to logic: if c.cacheBusting is true, we might generate new URL?
    // Just use a fixed TS or depend on upstream trigger?
    // The WC version uses Date.now() *at render time*.

    const imgPath = `/eros/cache/view_image?path=${encodeURIComponent(
      this.cachePath
    )}&subfolder=${encodeURIComponent(sub)}&filename=${encodeURIComponent(
      filename
    )}${ts}`;
    const tags = this.imageTags.get(base);

    return html`
      <div
        class="eros-item ${this.selectedFilename === f ||
        this.selectedFilename === base
          ? "selected"
          : ""}"
        @click=${() => this.select(f, imgPath)}
      >
        <div class="eros-loader"></div>
        <img
          src="${imgPath}"
          loading="lazy"
          style="opacity:0; transition:opacity 0.2s;"
          @load=${(e) => {
            e.target.style.opacity = "1";
            e.target.previousElementSibling.style.display = "none";
          }}
        />

        ${c.overlayEnabled
          ? html`
              <img
                class="eros-overlay"
                src="/eros/cache/view_image?path=${encodeURIComponent(
                  this.cachePath
                )}&subfolder=original&filename=${encodeURIComponent(
                  filename
                )}${ts}"
                style="opacity:${c.opacity}; mix-blend-mode:${c.blendMode}; position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"
                @error=${(e) => (e.target.style.display = "none")}
              />
            `
          : ""}
        ${c.showTagBadges && tags && tags.size > 0
          ? html`
              <div class="eros-tag-badges">
                ${Array.from(tags)
                  .slice(0, 6)
                  .map((t) => html`<span class="eros-tag-badge">${t}</span>`)}
                ${tags.size > 6
                  ? html`<span class="eros-tag-badge" style="background:#555;"
                      >+${tags.size - 6}</span
                    >`
                  : ""}
              </div>
            `
          : ""}

        <div class="eros-item-label">${base}</div>
      </div>
    `;
  }

  select(f, path) {
    // Remove .selected class from others? Lit render handles class binding if we track selection state.
    // But for simplicity/hybrid, we can just use DOM or track 'selected' prop.
    // Let's rely on parent to track selection or just use rudimentary DOM queries if needed,
    // OR better: track selectedFilename property.
    // Dispatch event
    this.dispatchEvent(
      new CustomEvent("image-selected", {
        detail: { filename: f, imgPath: path },
        bubbles: true,
        composed: true,
      })
    );
    // Emit a global event so listeners can react, but receivers should
    // filter events to only update when appropriate (linked to active node).
    try {
      window.dispatchEvent(
        new CustomEvent("eros.cache.image.selected", {
          detail: { filename: f, imgPath: path },
        })
      );
    } catch (e) {
      /* ignore */
    }
    // Local selection highlight?
    // const items = this.shadowRoot.querySelectorAll('.eros-item');
    // items.forEach(i => i.classList.remove('selected'));
    // e.currentTarget.classList.add('selected');
    // Doing this via data binding is cleaner in Lit.
    this.selectedFilename = f;
    this.requestUpdate();
  }

  updated(changed) {
    super.updated(changed);
    // Scroll to selected if changed
    if (changed.has("selectedFilename") || changed.has("files")) {
      if (this.selectedFilename) {
        // Defer slightly to ensure rendering
        setTimeout(() => {
          const selEl = this.shadowRoot.querySelector(".eros-item.selected");
          if (selEl) {
            selEl.scrollIntoView({ block: "center", behavior: "smooth" });
          }
        }, 100);
      }
    }
  }
}
customElements.define("eros-lit-grid", ErosLitGrid);

// ========================================================
// Lit Sidebar
// ========================================================
class ErosLitSidebar extends LitElement {
  static properties = {
    allTags: { type: Object }, // Map
    activeFilters: { type: Object }, // Set
    selectedTags: { type: Object }, // Set or null
    tagSearchQuery: { type: String },
    hasSelection: { type: Boolean },
    selectedFilename: { type: String },
    activeNode: { type: Object },
  };

  constructor() {
    super();
    this.activeFilters = new Set();
    this.collapsed = { filter: false, selected: false };
    this._activeNode = null;
    this._activeFilename = null;
    this._deleteAllFlag = false;
  }

  connectedCallback() {
    super.connectedCallback();
    // Listen for nodes opening the browser so we can highlight/link
    this._onBrowserOpen = (ev) => {
      try {
        const node = ev?.detail?.node;
        this._activeNode = node || null;
        this._activeFilename =
          node?.widgets?.find((w) => w.name === "filename")?.value || null;
        this.requestUpdate();
      } catch (e) {}
    };
    window.addEventListener("eros.cache.browser.open", this._onBrowserOpen);
  }

  disconnectedCallback() {
    try {
      if (this._onBrowserOpen)
        window.removeEventListener(
          "eros.cache.browser.open",
          this._onBrowserOpen
        );
    } catch (e) {}
    super.disconnectedCallback();
  }

  render() {
    const query = (this.tagSearchQuery || "").toLowerCase();

    // Prefer property-based linkage when available
    const linkedName =
      this.selectedFilename ||
      (this.activeNode &&
        this.activeNode.widgets?.find((w) => w.name === "filename")?.value) ||
      null;

    return html`
      <style>
        ${DRAWER_CSS}
      </style>
      <div class="eros-tag-sidebar">
        ${linkedName
          ? html`<div
              style="font-size:12px;color:#9ca2ad;padding:6px 0;border-bottom:1px solid #333;"
            >
              Linked: ${linkedName}
            </div>`
          : html`<div
              style="font-size:12px;color:#9ca2ad;padding:6px 0;border-bottom:1px solid #333;"
            >
              No node linked — open a node and click "Open Browser" to link
            </div>`}
        <!-- Filter Section -->
        <div class="eros-tag-section">
          <div
            class="eros-tag-section-label"
            @click=${() => {
              this.collapsed.filter = !this.collapsed.filter;
              this.requestUpdate();
            }}
          >
            Filter by Tags <span>${this.collapsed.filter ? "▶" : "▼"}</span>
          </div>
          <div
            class="eros-tag-section-content"
            style="display:${this.collapsed.filter ? "none" : "flex"}"
          >
            <input
              class="eros-tag-input"
              placeholder="Search tags..."
              @input=${(e) =>
                this.dispatchEvent(
                  new CustomEvent("filter-search", {
                    detail: e.target.value,
                    bubbles: true,
                    composed: true,
                  })
                )}
            />
            <div class="eros-tag-chips">
              ${Array.from(this.allTags || []).map(([name, count]) => {
                if (!name) return null;
                if (query && !name.toLowerCase().includes(query)) return null;
                return html`
                  <div
                    class="eros-tag-chip ${this.activeFilters.has(name)
                      ? "active"
                      : ""}"
                    @click=${(e) =>
                      this.dispatchEvent(
                        new CustomEvent("filter-click", {
                          detail: {
                            name,
                            ctrlKey: e.ctrlKey,
                            shiftKey: e.shiftKey,
                          },
                          bubbles: true,
                          composed: true,
                        })
                      )}
                  >
                    ${name}
                    <span style="opacity:0.6;font-size:9px;">${count}</span>
                  </div>
                `;
              })}
            </div>
          </div>
        </div>

        <!-- Selected Section -->
        <div class="eros-tag-section">
          <div
            class="eros-tag-section-label"
            @click=${() => {
              this.collapsed.selected = !this.collapsed.selected;
              this.requestUpdate();
            }}
          >
            Selected Tags <span>${this.collapsed.selected ? "▶" : "▼"}</span>
          </div>
          <div
            class="eros-tag-section-content"
            style="display:${this.collapsed.selected ? "none" : "flex"}"
          >
            ${!this.hasSelection
              ? html`<div style="font-size:10px; color:#666;">
                  Select an image...
                </div>`
              : html`
                  <div class="eros-tag-chips">
                    ${!this.selectedTags || this.selectedTags.size === 0
                      ? html`<div style="font-size:10px; color:#888;">
                          No tags
                        </div>`
                      : Array.from(this.selectedTags).map(
                          (t) => html`
                            <div class="eros-tag-chip">
                              ${t}
                              <span
                                class="eros-tag-chip-remove"
                                @click=${(e) => {
                                  e.stopPropagation();
                                  this.dispatchEvent(
                                    new CustomEvent("tag-remove", {
                                      detail: t,
                                      bubbles: true,
                                      composed: true,
                                    })
                                  );
                                }}
                                >×</span
                              >
                            </div>
                          `
                        )}
                  </div>
                  <div class="eros-tag-input-row">
                    <input
                      class="eros-tag-input"
                      id="inp-new"
                      placeholder="Add tag..."
                      @keydown=${(e) => {
                        if (e.key === "Enter")
                          this._addTag(e.target.value, e.target);
                      }}
                    />
                    <button
                      class="eros-tag-add-btn"
                      @click=${(e) =>
                        this._addTag(
                          this.shadowRoot.getElementById("inp-new").value,
                          this.shadowRoot.getElementById("inp-new")
                        )}
                    >
                      +
                    </button>
                    <button
                      class="eros-tag-add-btn"
                      @click=${() =>
                        this.dispatchEvent(
                          new CustomEvent("tag-auto", {
                            bubbles: true,
                            composed: true,
                          })
                        )}
                    >
                      🤖
                    </button>
                  </div>
                  <div
                    style="display:flex; flex-direction:column; gap:6px; margin-top:8px;"
                  >
                    <label
                      style="font-size:12px; display:flex; align-items:center; gap:8px;"
                    >
                      <input
                        type="checkbox"
                        id="del-all"
                        @click=${(e) => {
                          e.stopPropagation();
                          this._deleteAllFlag = e.target.checked;
                        }}
                      />
                      Delete all maps for original
                    </label>
                    <div>
                      <button
                        class="eros-btn"
                        style="background:#8b2a2a; color:white;"
                        @click=${(e) => {
                          e.stopPropagation();
                          this._onDelete();
                        }}
                      >
                        Delete Map
                      </button>
                    </div>
                  </div>
                `}
          </div>
        </div>
      </div>
    `;
  }

  _addTag(val, input) {
    if (!val || !val.trim()) return;
    this.dispatchEvent(
      new CustomEvent("tag-add", {
        detail: val.trim(),
        bubbles: true,
        composed: true,
      })
    );
    if (input) input.value = "";
  }

  _onDelete() {
    // Dispatch an event upward so the orchestrator can perform deletion
    this.dispatchEvent(
      new CustomEvent("image-delete", {
        detail: { all: !!this._deleteAllFlag },
        bubbles: true,
        composed: true,
      })
    );
  }
}
customElements.define("eros-lit-sidebar", ErosLitSidebar);

// ========================================================
// Lit Orchestrator
// ========================================================
export class ErosLitBrowser extends LitElement {
  static properties = {
    isOpen: { type: Boolean },
    settings: { type: Object },
    files: { type: Array },
    activeFilters: { type: Object }, // Set
    tagSearchQuery: { type: String },
    currentTab: { type: String },
    selectedFilename: { type: String },
  };

  constructor() {
    super();
    this.cache = new CacheService();
    this.files = [];
    this.activeFilters = new Set();
    this.settings = {};
    this.isOpen = false;

    // Load settings
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
      currentTab: "depth",
    };
    try {
      this.settings = {
        ...defaults,
        ...JSON.parse(
          localStorage.getItem("eros_cache_browser_settings") || "{}"
        ),
      };
    } catch {
      this.settings = defaults;
    }

    this.currentTab = this.settings.currentTab;

    // Bind cache events to update UI
    this.cache.subscribe((evt, data) => {
      if (
        evt === "tag-added" ||
        evt === "tag-removed" ||
        evt === "tags-loaded" ||
        evt === "map-deleted"
      ) {
        this.requestUpdate();
      }
    });
  }

  saveSettings() {
    try {
      localStorage.setItem(
        "eros_cache_browser_settings",
        JSON.stringify(this.settings)
      );
    } catch {}
  }

  connectedCallback() {
    super.connectedCallback();
    this.cache.setCachePath(this.dataset.cachePath || ""); // Passed from parent via attribute?
    console.log(
      "[Lit] connectedCallback; dataset.cachePath=",
      this.dataset.cachePath,
      "_sidebarMethod=",
      this._sidebarMethod
    );
    this.cache.loadTags();
    // Default to 'original' when opened manually (no linked node)
    if (!this.currentTab) this.currentTab = "original";
    try {
      this.fetchFiles(false);
    } catch (e) {}
  }

  // API
  open(node) {
    console.log("[Lit] Opening for node:", node);
    this.activeNode = node;
    this.isOpen = true;
    this.cache.setCachePath(
      node?.widgets?.find((w) => w.name === "cache_path")?.value || ""
    );

    // Initial Selection Logic
    if (node) {
      const wVal = node.widgets?.find((w) => w.name === "filename")?.value;
      if (wVal) {
        // filename might be 'folder/image.png' or just 'image.png'
        // We want to extract the basename and/or sync the tab?
        // For now, let's try to match the filename.
        // If it contains a slash, the first part might be the tab.
        const parts = wVal.split("/");
        if (parts.length > 1) {
          const tab = parts[0];
          if (MAP_TABS.includes(tab)) {
            this.currentTab = tab;
            this.settings = { ...this.settings, currentTab: tab };
          }
          this.selectedFilename = parts[1]; // just the name
        } else {
          this.selectedFilename = wVal;
        }
      }
    }

    this.fetchFiles(false);
  }
  close() {
    this.isOpen = false;
  }

  // Helper: Update the actual node widget/preview
  _updateNode(filename, imgPath) {
    if (!this.activeNode) return;
    const w = this.activeNode.widgets?.find((w) => w.name === "filename");
    if (w) {
      console.log("[Lit] Node Update Triggered:", filename);
      // If filename already includes a subfolder prefix, use it as-is.
      if (filename && filename.includes("/")) {
        w.value = filename;
      } else {
        w.value = `${this.currentTab}/${filename}`;
      }
      w.callback?.(w.value);

      // Image Preview (Canvas)
      if (!imgPath) {
        // Construct if missing (e.g. auto-update from tab switch) ...
      }
      // Avoid re-loading if same? No, user might want to simple click to refresh.

      // Only update if value actually changed or forced?
      // In Comfy, assigning imgs always triggers redraw. That's fine.

      // But let's log it to be sure.
      // console.log("Updating node activeNode:", this.activeNode);

      const img = new Image();
      img.onload = () => {
        this.activeNode.imgs = [img];
        app.graph.setDirtyCanvas(true);
      };
      img.src = imgPath;
    }
  }

  async fetchFiles(updateNode = false, forceAll = false) {
    // If no cache path is configured (manual sidebar open), fetch across
    // all known map types so the user sees all maps. Otherwise fetch for
    // the currentTab only.
    const MAP_TYPES = [
      "depth",
      "canny",
      "openpose",
      "lineart",
      "scribble",
      "softedge",
      "normal",
      "seg",
      "shuffle",
      "mediapipe_face",
      "custom",
    ];

    let combined = [];
    try {
      if (!this.cache.cachePath && !forceAll) {
        // If no cachePath and user selected a specific tab, fetch that tab
        if (this.currentTab) {
          const f = await this.cache.fetchFiles(this.currentTab);
          combined = f.map((p) => `${this.currentTab}/${p}`);
        } else {
          // No tab selected: fetch all types
          for (const t of MAP_TYPES) {
            try {
              const parts = await this.cache.fetchFiles(t);
              if (parts && parts.length)
                combined.push(...parts.map((p) => `${t}/${p}`));
            } catch (e) {
              /* ignore per-type failures */
            }
          }
        }
      } else if (!this.cache.cachePath && forceAll) {
        for (const t of MAP_TYPES) {
          try {
            const parts = await this.cache.fetchFiles(t);
            if (parts && parts.length)
              combined.push(...parts.map((p) => `${t}/${p}`));
          } catch (e) {
            /* ignore per-type failures */
          }
        }
      } else {
        const f = await this.cache.fetchFiles(this.currentTab);
        combined = f.map((p) => `${this.currentTab}/${p}`);
      }
    } catch (e) {
      combined = [];
    }

    this.files = combined;

    // Restore selection logic
    if (this.selectedFilename) {
      const base = this.cache.getBasename(this.selectedFilename);
      const match = this.files.find(
        (file) => this.cache.getBasename(file) === base
      );
      if (match) {
        this.selectedFilename = match;
        if (updateNode) this._updateNode(match);
      } else this.selectedFilename = null;
    }

    // Load tags for each file entry (strip any prefixed subfolder)
    this.files.forEach((file) => {
      const name =
        file && file.includes("/") ? file.split("/").slice(1).join("/") : file;
      const base = this.cache.getBasename(name);
      if (!this.cache.imageTags.has(base)) this.cache.loadImageTags(base);
    });
  }

  firstUpdated() {
    // Resize handled by ComfyUI sidebar now — no-op.
  }

  render() {
    // Filter logic
    let displayFiles = this.files;
    const q = (this.tagSearchQuery || "").toLowerCase();
    if (q || this.activeFilters.size > 0) {
      displayFiles = displayFiles.filter((f) => {
        const base = this.cache.getBasename(f);
        const tags = this.cache.imageTags.get(base);
        if (!tags && (q || this.activeFilters.size > 0)) {
          if (this.activeFilters.size > 0) return false;
        } // no tags loaded yet

        if (this.activeFilters.size > 0) {
          for (let ft of this.activeFilters)
            if (!tags || !tags.has(ft)) return false;
        }
        if (q) {
          if (base.toLowerCase().includes(q)) return true;
          if (tags) {
            for (let t of tags) if (t.toLowerCase().includes(q)) return true;
          }
          return false;
        }
        return true;
      });
    }

    const selectedTags = this.selectedFilename
      ? this.cache.imageTags.get(this.cache.getBasename(this.selectedFilename))
      : null;

    return html`
      <style>
        ${DRAWER_CSS} ${BTN_CSS} .eros-wc-container {
          display: flex;
          height: 100%;
          overflow: hidden;
          width: 100%;
        }
        :host {
          max-height: 100%;
          width: 100%;
          display: block;
          box-sizing: border-box;
        }
        .eros-drawer {
          max-height: 100%;
          width: 100%;
          box-sizing: border-box;
        }
        .eros-main-column {
          display: flex;
          flex-direction: column;
          flex: 1;
          overflow: hidden;
        }
        eros-lit-controls {
          flex-shrink: 0;
        }
        eros-lit-grid {
          flex: 1;
          overflow: hidden;
        }
        eros-lit-sidebar {
          width: 300px;
          border-left: 1px solid #333;
          background: rgba(0, 0, 0, 0.2);
          flex-shrink: 0;
        }
      </style>

      <div class="eros-drawer ${this.isOpen ? "open" : ""}" id="drawer">
        <!-- resize handle removed: sizing controlled by ComfyUI sidebar -->

        <div class="eros-drawer-header">
          <div style="display:flex;align-items:center;gap:10px;">
            <h3>
              Cache Browser
              ${this.activeNode
                ? html` — Node:
                  ${this.activeNode.id ||
                  this.activeNode.uid ||
                  this.activeNode.name ||
                  "(unknown)"}`
                : ""}
            </h3>
            <button
              class="eros-btn-small"
              style="font-size:12px;padding:2px 8px;"
              @click=${() => app.queuePrompt(0, 1)}
            >
              Run ▶
            </button>
          </div>
          <!-- close button removed: sidebar icon handles toggling -->
        </div>

        <div class="eros-wc-container">
          <div class="eros-main-column">
            <eros-lit-controls
              .config=${this.settings}
              @setting-change=${this._handleSetting}
              @tab-changed=${this._handleTab}
              @refresh-requested=${() => this.fetchFiles(false)}
            ></eros-lit-controls>

            <eros-lit-grid
              .files=${displayFiles}
              .imageTags=${new Map(this.cache.imageTags)}
              .cachePath=${this.cache.cachePath}
              .currentTab=${this.currentTab}
              .config=${this.settings}
              .selectedFilename=${this.selectedFilename}
              @image-selected=${(e) => {
                // Filename may be prefixed with subfolder (type/file.png)
                const fn = e.detail.filename;
                this.selectedFilename = fn;
                this._updateNode(fn, e.detail.imgPath);
              }}
            ></eros-lit-grid>
          </div>

          <eros-lit-sidebar
            .allTags=${new Map(this.cache.allTags)}
            .activeFilters=${this.activeFilters}
            .selectedTags=${selectedTags ? new Set(selectedTags) : null}
            .tagSearchQuery=${this.tagSearchQuery}
            .hasSelection=${!!this.selectedFilename}
            .selectedFilename=${this.selectedFilename}
            .activeNode=${this.activeNode}
            @filter-search=${(e) => {
              this.tagSearchQuery = e.detail;
            }}
            @filter-click=${(e) => {
              if (e.detail.ctrlKey) {
                if (this.activeFilters.has(e.detail.name))
                  this.activeFilters.delete(e.detail.name);
                else this.activeFilters.add(e.detail.name);
                this.activeFilters = new Set(this.activeFilters);
              } else if (e.detail.shiftKey) {
                if (this.selectedFilename) {
                  const base = this.cache.getBasename(this.selectedFilename);
                  this.cache.addTag(base, e.detail.name);
                }
              } else {
                this.activeFilters.clear();
                this.activeFilters.add(e.detail.name);
                this.activeFilters = new Set(this.activeFilters);
              }
            }}
            @tag-add=${(e) =>
              this.cache.addTag(
                this.cache.getBasename(this.selectedFilename),
                e.detail
              )}
            @tag-remove=${(e) =>
              this.cache.removeTag(
                this.cache.getBasename(this.selectedFilename),
                e.detail
              )}
            @tag-auto=${() =>
              this.cache.autoTag(this.cache.getBasename(this.selectedFilename))}
            @image-delete=${(e) =>
              this._handleImageDelete(e.detail && e.detail.all)}
          ></eros-lit-sidebar>
        </div>
      </div>
      ${this.isOpen
        ? html`<div
            class="eros-modal-bg"
            style="display:block;"
            @click=${this.close}
          ></div>`
        : ""}
    `;
  }

  _handleSetting(e) {
    this.settings = { ...this.settings, [e.detail.key]: e.detail.value };
    this.saveSettings();
  }
  _handleTab(e) {
    const tab = e?.detail?.tab || e?.detail || e;
    this.currentTab = tab;
    this.settings = { ...this.settings, currentTab: tab };
    // If a node is linked, fetch and update the node selection if a matching
    // map exists in the newly selected tab. Otherwise just refresh the list.
    try {
      const updateNode = !!this.activeNode;
      this.fetchFiles(updateNode);
    } catch (ex) {
      // ignore
    }
  }

  async _handleImageDelete(allFlag) {
    const basename = this.cache.getBasename(this.selectedFilename || "");
    if (!basename) return;

    // Confirm with user (basic confirm dialog)
    try {
      const ok = confirm(
        allFlag
          ? `Delete ALL maps for '${basename}' and remove its tags?`
          : `Delete map '${this.selectedFilename}' and remove its tags?`
      );
      if (!ok) return;
    } catch (e) {}

    const res = await this.cache.deleteMap(
      basename,
      this.currentTab,
      this.cache.cachePath,
      !!allFlag
    );
    if (res && res.success) {
      // Clear selection and refresh
      this.selectedFilename = null;
      // If an active node references this basename, clear its filename
      try {
        const node = this.activeNode;
        if (node) {
          const w = node.widgets?.find((w) => w.name === "filename");
          if (w) {
            const val = w.value || "";
            const valBasename = this.cache.getBasename(val);
            if (valBasename === basename) {
              w.value = "";
              w.callback?.(w.value);
              // clear preview
              node.imgs = [];
              app.graph.setDirtyCanvas(true);
            }
          }
        }
      } catch (e) {}

      // Reload files and tags
      try {
        await this.fetchFiles(false, true);
        this.cache.loadTags();
      } catch (e) {}
    } else {
      console.error("Delete failed", res);
      try {
        alert("Delete failed: " + (res && (res.error || res.message)));
      } catch (e) {}
    }
  }

  setActiveNode(node) {
    // Assign active node and sync cache path + initial selection
    this.activeNode = node;
    this.cache.setCachePath(
      node?.widgets?.find((w) => w.name === "cache_path")?.value || ""
    );

    if (node) {
      const wVal = node.widgets?.find((w) => w.name === "filename")?.value;
      if (wVal) {
        const parts = wVal.split("/");
        if (parts.length > 1) {
          const tab = parts[0];
          if (MAP_TABS.includes(tab)) {
            this.currentTab = tab;
            this.settings = { ...this.settings, currentTab: tab };
          }
          this.selectedFilename = `${tab}/${parts.slice(1).join("/")}`;
        } else {
          this.selectedFilename = wVal;
        }
      }
    }
    this.requestUpdate();
  }

  open(node) {
    console.log(
      "[Lit] Opening for node:",
      node,
      "_sidebarMethod=",
      this._sidebarMethod,
      "_sidebarId=",
      this._sidebarId
    );
    // Ensure active node is set first so header and state update
    if (node) this.setActiveNode(node);
    this.isOpen = true;
    // If opening for a node, try to select matching file and update node
    try {
      this.fetchFiles(!!node);
    } catch (e) {}
  }
}

// Register the main browser element
try {
  customElements.define("eros-lit-browser", ErosLitBrowser);
} catch (ex) {
  // ignore if already defined
}
