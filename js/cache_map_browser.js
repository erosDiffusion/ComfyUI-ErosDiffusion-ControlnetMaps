/**
 * Author: ErosDiffusion (EF)
 * Email: erosdiffusionai+controlnetmaps@gmail.com
 * Year: 2025
 */

import { app } from "../../scripts/app.js";
// Simplified: always use the Lit implementation and remove preference switch.

// Register the persistent sidebar tab at module load so it's independent
// from node registration. This makes the sidebar a standalone component
// that nodes can link to via global events.
// Register the persistent sidebar tab at module load so it's independent
// from node registration. The sidebar is registered immediately; the
// Lit implementation is lazy-loaded inside the `render` callback so the
// registration itself is synchronous and guaranteed to happen once.
(function initPersistentSidebar() {
  try {
    if (
      app?.extensionManager &&
      typeof app.extensionManager.registerSidebarTab === "function"
    ) {
      const sid = "eros-cache-sidebar";
      try {
        const sidebarCleanup = app.extensionManager.registerSidebarTab({
          id: sid,
          icon: "pi pi-folder",
          title: "Cache Browser",
          tooltip: "Cache Browser",
          type: "custom",
          render: (containerEl) => {
            containerEl.style.padding = "0";
            // Start lazy import and mount the full browser when ready.
            (async () => {
              try {
                await import("./cache_map_browser_lit.js");
                let bEl = window._eros_cache_browser;
                if (!bEl) {
                  bEl = document.createElement("eros-lit-browser");
                  // Mark the element with sidebar metadata so nodes can interact
                  try {
                    bEl._sidebarMethod = "extensionManager.registerSidebarTab";
                    bEl._sidebarId = sid;
                    bEl._sidebar = app.extensionManager;
                  } catch (e) {}
                  window._eros_cache_browser = bEl;
                }
                if (bEl.parentNode !== containerEl)
                  containerEl.appendChild(bEl);
                console.log(
                  "[CacheMapBrowser] mounted eros-lit-browser (module init)"
                );
              } catch (err) {
                console.warn(
                  "[CacheMapBrowser] lazy import/mount failed in render:",
                  err
                );
              }
            })();

            return () => {
              try {
                const bEl = window._eros_cache_browser;
                if (bEl && bEl.parentNode) bEl.parentNode.removeChild(bEl);
              } catch (e) {}
            };
          },
        });
        console.log(
          "[CacheMapBrowser] registered persistent sidebar at module init id=",
          sid,
          "cleanup=",
          typeof sidebarCleanup === "function"
        );
      } catch (ex) {
        console.warn(
          "[CacheMapBrowser] failed to register persistent sidebar at module init:",
          ex
        );
      }
    }
  } catch (ex) {
    /* ignore */
  }
})();

app.registerExtension({
  name: "ErosDiffusion.CacheMapBrowser",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "CacheMapBrowserNode") {
      // Helper to load and create the lit drawer (simple: create the element and append to body)
      let drawerInstance = null;
      const getDrawer = async () => {
        if (drawerInstance) return drawerInstance;
        // If a persistent sidebar instance was already registered and mounted,
        // reuse that instead of creating a new fallback instance.
        if (window._eros_cache_browser) {
          console.log(
            "[CacheMapBrowser] reusing registered window._eros_cache_browser"
          );
          drawerInstance = window._eros_cache_browser;
          return drawerInstance;
        }
        try {
          console.log(
            "[CacheMapBrowser] importing lit module ./cache_map_browser_lit.js..."
          );
          await import("./cache_map_browser_lit.js");
          const browserEl = document.createElement("eros-lit-browser");
          // Expose globally for debug and external consumers
          try {
            window._eros_cache_browser = browserEl;
          } catch (e) {}

          // Default behavior: append to document.body. The persistent sidebar
          // (eros-cache-sidebar) is registered at module init and will handle
          // embedded sidebar rendering; nodes should only open/show that
          // sidebar via the global open event.
          document.body.appendChild(browserEl);
          console.log(
            "[CacheMapBrowser] created eros-lit-browser and appended to document.body"
          );
          drawerInstance = browserEl;
          return browserEl;
        } catch (e) {
          console.error("Failed to create CacheMap Browser (lit):", e);
          return null;
        }
      };

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined;

        this.addWidget("button", "Open Browser", "open", async () => {
          if (!this.drawer) this.drawer = await getDrawer();
          if (!this.drawer) return;

          // If the component was registered with the sidebar/extensionManager, try to open/show it via sidebar APIs
          const sidebar =
            this.drawer._sidebar || app?.ui?.sidebar || app?.extensionManager;
          console.log(
            "[CacheMapBrowser] attempting to open sidebar. sidebar present:",
            !!sidebar,
            "drawer._sidebarMethod:",
            this.drawer._sidebarMethod,
            "drawer._sidebarId:",
            this.drawer._sidebarId
          );
          // Broadcast that a node opened the browser so standalone sidebar can react
          try {
            window.dispatchEvent(
              new CustomEvent("eros.cache.browser.open", {
                detail: { node: this },
              })
            );
          } catch (e) {}
          if (sidebar) {
            try {
              const id = this.drawer._sidebarId || "eros-cache-sidebar";
              if (typeof sidebar.openTab === "function") {
                console.log(
                  "[CacheMapBrowser] calling sidebar.openTab(%s)",
                  id
                );
                sidebar.openTab(id);
                try {
                  this.drawer.open(this);
                } catch (e) {}
                return;
              } else if (typeof sidebar.open === "function") {
                console.log("[CacheMapBrowser] calling sidebar.open(%s)", id);
                sidebar.open(id);
                try {
                  this.drawer.open(this);
                } catch (e) {}
                return;
              } else if (typeof sidebar.showTab === "function") {
                console.log(
                  "[CacheMapBrowser] calling sidebar.showTab(%s)",
                  id
                );
                sidebar.showTab(id);
                try {
                  this.drawer.open(this);
                } catch (e) {}
                return;
              } else if (typeof sidebar.activateTab === "function") {
                console.log(
                  "[CacheMapBrowser] calling sidebar.activateTab(%s)",
                  id
                );
                sidebar.activateTab(id);
                try {
                  this.drawer.open(this);
                } catch (e) {}
                return;
              } else if (typeof sidebar.show === "function") {
                console.log("[CacheMapBrowser] calling sidebar.show(%s)", id);
                sidebar.show(id);
                try {
                  this.drawer.open(this);
                } catch (e) {}
                return;
              }
              console.log(
                "[CacheMapBrowser] sidebar found but no known open method matched, will fallback to component.open"
              );
            } catch (e) {
              console.warn(
                "[CacheMapBrowser] Sidebar open API failed, falling back to component.open:",
                e
              );
            }
          }

          // Fallback: call the component's open method
          if (typeof this.drawer.open === "function") this.drawer.open(this);

          // The browser instance updates the active (linked) node directly
          // when an image is selected. No global selection listeners are
          // required on the node side.
        });

        this.addWidget("button", "Run", "run", () => {
          app.queuePrompt(0, 1);
        });

        // Register a filtered global listener so this node only responds
        // to selections when it is the browser's active node.
        try {
          const handler = (ev) => {
            const d = ev.detail || {};
            const filename = d.filename || null;
            if (!filename) return;
            try {
              const browser = window._eros_cache_browser;
              if (!browser || browser.activeNode !== this) return;
            } catch (e) {
              return;
            }
            const w = this.widgets?.find((w) => w.name === "filename");
            if (w) {
              const browser = window._eros_cache_browser;
              // If filename already includes a subfolder prefix, use it
              if (filename.includes("/")) {
                w.value = filename;
              } else {
                w.value = `${
                  (browser && browser.currentTab) || ""
                }/${filename}`;
              }
              w.callback?.(w.value);
              if (d.imgPath) {
                try {
                  this.imgs = [new Image()];
                  this.imgs[0].src = d.imgPath;
                  app.graph.setDirtyCanvas(true);
                } catch (e) {}
              }
            }
          };
          if (this._eros_cache_handler)
            window.removeEventListener(
              "eros.cache.image.selected",
              this._eros_cache_handler
            );
          this._eros_cache_handler = handler;
          window.addEventListener("eros.cache.image.selected", handler);
        } catch (e) {}

        return r;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;

        // On initial load, if we have a filename, try to preload the thumbnail
        setTimeout(() => {
          const fileWidget = this.widgets?.find((w) => w.name === "filename");
          const cacheWidget = this.widgets?.find(
            (w) => w.name === "cache_path"
          );
          // stray append removed here; getDrawer already handles appending when needed
          if (fileWidget && fileWidget.value) {
            const parts = fileWidget.value.split("/");
            if (parts.length >= 2) {
              const sub = parts[0];
              const file = parts.slice(1).join("/");
              const cache = cacheWidget ? cacheWidget.value : "";

              const imgPath = `/eros/cache/view_image?path=${encodeURIComponent(
                cache
              )}&subfolder=${encodeURIComponent(
                sub
              )}&filename=${encodeURIComponent(file)}&t=${Date.now()}`;

              const img = new Image();
              img.onload = () => {
                this.imgs = [img];
                app.graph.setDirtyCanvas(true);
              };
              img.src = imgPath;
            }
          }
        }, 100);

        return r;
      };
    }
  },
});
