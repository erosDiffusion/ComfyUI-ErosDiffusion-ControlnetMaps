/**
 * Author: ErosDiffusion (EF)
 * Email: erosdiffusionai+controlnetmaps@gmail.com
 * Year: 2025
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
// Simplified: always use the Lit implementation and remove preference switch.

// ComfyUI comms bridge (server -> frontend)
// Register once at module load so `PromptServer.instance.send_sync("eros.*", ...)`
// messages are handled even if the sidebar UI has not been opened yet.
(function initErosCommsBridge() {
  try {
    if (window.__eros_comms_bridge_initialized) return;
    window.__eros_comms_bridge_initialized = true;

    if (!api || typeof api.addEventListener !== "function") return;

    const forward = (type) => {
      try {
        api.addEventListener(type, (ev) => {
          try {
            const payload = (ev && (ev.detail || ev.data)) || ev;
            window.dispatchEvent(new CustomEvent(type, { detail: payload }));
          } catch (e) {}
        });
      } catch (e) {}
    };

    [
      "eros.tags.updated",
      "eros.image.deleted",
      "eros.map.saved",
      "eros.image.saved",
      "eros.cache.imported",
      "eros.cache.reset",
    ].forEach(forward);
  } catch (e) {
    // ignore
  }
})();

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
          title: "Controlnet Map Browser",
          tooltip: "Controlnet Map Browser",
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
  // About panel badges (appears in ComfyUI Settings -> About)
  aboutPageBadges: [
    {
      label: "Controlnet Map Browser",
      url: "https://github.com/erosDiffusion/ErosDiffusion-ControlnetMaps",
      icon: "pi pi-github",
    },
    {
      label: "Donate",
      url: "https://donate.stripe.com/3cI7sDgZg4rr2Ln0HfcV202",
      icon: "pi pi-heart",
    },
  ],
  // Selection toolbox commands: adds a button when nodes are selected
  commands: [
    {
      id: "eros-cache.open-connect",
      label: "Open/Connect Map Browser",
      icon: "pi pi-folder",
      function: async (selectedItem) => {
        try {
          // Ensure browser element exists
          let browser = window._eros_cache_browser;
          if (!browser) {
            try {
              await import("./cache_map_browser_lit.js");
            } catch (e) {}
            browser = window._eros_cache_browser;
            if (!browser) {
              try {
                browser = document.createElement("eros-lit-browser");
                window._eros_cache_browser = browser;
                document.body.appendChild(browser);
              } catch (e) {}
            }
          }

          // Dispatch open event and call open() if available
          try {
            window.dispatchEvent(
              new CustomEvent("eros.cache.browser.open", {
                detail: { node: selectedItem },
              })
            );
          } catch (e) {}
          try {
            if (browser && typeof browser.open === "function")
              browser.open(selectedItem);
          } catch (e) {}
        } catch (e) {
          console.error("eros-cache: open-connect command failed", e);
        }
      },
    },
  ],
  getSelectionToolboxCommands: (selectedItem) => {
    try {
      const selectedItems = app.canvas.selectedItems;
      const itemCount = selectedItems ? selectedItems.size : 0;
      // Show when exactly one node is selected
      if (itemCount === 1) return ["eros-cache.open-connect"];
    } catch (e) {}
    return [];
  },
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

        this.addWidget(
          "button",
          "Open/Connect to Map browser",
          "open",
          async () => {
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

            // Ensure the drawer component is connected to this node (connect event)
            try {
              if (typeof this.drawer.open === "function")
                this.drawer.open(this);
            } catch (e) {}

            if (sidebar) {
              try {
                const id = this.drawer._sidebarId || "eros-cache-sidebar";
                // Determine label for DOM lookup
                const label = "Controlnet Map Browser";
                // Check DOM button state: if the sidebar button is already selected/open,
                // avoid executing the toggle command which would close it.
                try {
                  const btn = document.querySelector(
                    `button[aria-label="${label}"]`
                  );
                  if (
                    btn &&
                    btn.classList.contains("side-bar-button-selected")
                  ) {
                    console.log(
                      "[CacheMapBrowser] sidebar button already selected; will not toggle"
                    );
                    try {
                      this.drawer.open(this);
                    } catch (e) {}
                    return;
                  }
                } catch (e) {}

                // Try ComfyUI command API to toggle the sidebar tab (preferred)
                try {
                  if (
                    app?.extensionManager?.command &&
                    typeof app.extensionManager.command.execute === "function"
                  ) {
                    const cmd = `Workspace.ToggleSidebarTab.${id}`;
                    console.log(
                      "[CacheMapBrowser] executing command to toggle sidebar tab:",
                      cmd
                    );
                    try {
                      app.extensionManager.command.execute(cmd);
                      try {
                        this.drawer.open(this);
                      } catch (e) {}
                      return;
                    } catch (e) {
                      // ignore and continue to other methods
                    }
                  }
                } catch (e) {}
                // Try common API names first
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

                // Try any method with an 'open'/'show'/'activate'/'select' name
                try {
                  const tryNames = Object.getOwnPropertyNames(sidebar || {});
                  for (const n of tryNames) {
                    try {
                      if (
                        typeof sidebar[n] === "function" &&
                        /open|show|activate|select/i.test(n)
                      ) {
                        console.log(
                          "[CacheMapBrowser] calling sidebar.%s(%s)",
                          n,
                          id
                        );
                        try {
                          sidebar[n](id);
                        } catch (err) {
                          // some methods may expect no args
                          try {
                            sidebar[n]();
                          } catch (err2) {}
                        }
                        try {
                          this.drawer.open(this);
                        } catch (e) {}
                        return;
                      }
                    } catch (err) {}
                  }
                } catch (err) {}

                console.log(
                  "[CacheMapBrowser] sidebar found but no known open method matched, already attempted drawer.open fallback"
                );
              } catch (e) {
                console.warn(
                  "[CacheMapBrowser] Sidebar open API failed, falling back to component.open:",
                  e
                );
              }
            }

            // Final fallback: ensure drawer is opened (connect) — already attempted above
            try {
              if (typeof this.drawer.open === "function")
                this.drawer.open(this);
            } catch (e) {}

            // The browser instance updates the active (linked) node directly
            // when an image is selected. No global selection listeners are
            // required on the node side.
          }
        );

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
