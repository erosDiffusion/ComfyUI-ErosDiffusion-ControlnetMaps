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
          ${[
            "original",
            "depth",
            "canny",
            "pose",
            "segmentation",
            "lineart",
            "openpose",
            "scribble",
            "softedge",
          ].map(
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
        }
      </style>
      <div class="eros-grid">
        ${this.files.map((f) => this.renderItem(f, c))}
      </div>
    `;
  }

  renderItem(f, c) {
    const base = f
      .split("/")
      .pop()
      .replace(/\.[^/.]+$/, "");
    const ts = c.cacheBusting ? `&t=${Date.now()}` : ""; // Lit handles re-render intelligently
    // Note: Using Date.now() in render() forces reload on every update?
    // Better to store timestamp in state if we want to control it, implies simple cache busting might cause reloads.
    // For now adhering to logic: if c.cacheBusting is true, we might generate new URL?
    // Just use a fixed TS or depend on upstream trigger?
    // The WC version uses Date.now() *at render time*.

    const imgPath = `/eros/cache/view_image?path=${encodeURIComponent(
      this.cachePath
    )}&subfolder=${this.currentTab}&filename=${encodeURIComponent(f)}${ts}`;
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
                )}&subfolder=original&filename=${encodeURIComponent(f)}${ts}"
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
  };

  constructor() {
    super();
    this.activeFilters = new Set();
    this.collapsed = { filter: false, selected: false };
  }

  render() {
    const query = (this.tagSearchQuery || "").toLowerCase();

    return html`
      <style>
        ${DRAWER_CSS}
      </style>
      <div class="eros-tag-sidebar">
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
        evt === "tags-loaded"
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
    this.cache.loadTags();
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
          if (
            [
              "depth",
              "canny",
              "pose",
              "segmentation",
              "lineart",
              "openpose",
              "scribble",
              "softedge",
              "original",
            ].includes(tab)
          ) {
            this.currentTab = tab;
            this.settings.currentTab = tab;
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
      w.value = `${this.currentTab}/${filename}`;
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

  async fetchFiles(updateNode = false) {
    const f = await this.cache.fetchFiles(this.currentTab);
    this.files = f;

    // Restore selection logic
    if (this.selectedFilename) {
      const base = this.cache.getBasename(this.selectedFilename);
      const match = f.find((file) => this.cache.getBasename(file) === base);
      if (match) {
        this.selectedFilename = match;
        // If we switched tabs and found a match, update the node
        if (updateNode) this._updateNode(match);
      } else this.selectedFilename = null;
    }

    f.forEach((file) => {
      const base = this.cache.getBasename(file);
      if (!this.cache.imageTags.has(base)) this.cache.loadImageTags(base);
    });
  }

  firstUpdated() {
    // Resize Handler Logic
    const handle = this.shadowRoot.querySelector(".eros-drawer-resize-handle");
    const drawer = this.shadowRoot.getElementById("drawer");

    if (handle && drawer) {
      let isResizing = false;
      let startX, startWidth;

      handle.addEventListener("mousedown", (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = drawer.getBoundingClientRect().width;
        document.body.style.cursor = "ew-resize";
        e.preventDefault(); // Prevent text selection
      });

      document.addEventListener("mousemove", (e) => {
        if (!isResizing) return;
        const newWidth = startWidth - (e.clientX - startX);
        if (newWidth > 300 && newWidth < 1200) {
          drawer.style.width = newWidth + "px";
        }
      });

      document.addEventListener("mouseup", () => {
        isResizing = false;
        document.body.style.cursor = "";
      });
    }
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
        <div class="eros-drawer-resize-handle"></div>

        <div class="eros-drawer-header">
          <div style="display:flex;align-items:center;gap:10px;">
            <h3>Cache Browser</h3>
            <button
              class="eros-btn-small"
              style="font-size:12px;padding:2px 8px;"
              @click=${() => app.queuePrompt(0, 1)}
            >
              Run ▶
            </button>
          </div>
          <div class="eros-drawer-close" @click=${this.close}>×</div>
        </div>

        <div class="eros-wc-container">
          <div class="eros-main-column">
            <eros-lit-controls
              .config=${this.settings}
              @setting-change=${this._handleSetting}
              @tab-changed=${this._handleTab}
              @refresh-requested=${this.fetchFiles}
            ></eros-lit-controls>

            <eros-lit-grid
              .files=${displayFiles}
              .imageTags=${new Map(this.cache.imageTags)}
              .cachePath=${this.cache.cachePath}
              .currentTab=${this.currentTab}
              .config=${this.settings}
              .selectedFilename=${this.selectedFilename}
              @image-selected=${(e) => {
                this.selectedFilename = e.detail.filename;
                this._updateNode(e.detail.filename, e.detail.imgPath);
              }}
            ></eros-lit-grid>
          </div>

          <eros-lit-sidebar
            .allTags=${new Map(this.cache.allTags)}
            .activeFilters=${this.activeFilters}
            .selectedTags=${selectedTags ? new Set(selectedTags) : null}
            .tagSearchQuery=${this.tagSearchQuery}
            .hasSelection=${!!this.selectedFilename}
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
    this.currentTab = e.detail.tab;
    this.settings.currentTab = e.detail.tab;
    this.saveSettings();
    this.fetchFiles(true); // Propagate selection update to node
  }
}
customElements.define("eros-lit-browser", ErosLitBrowser);
