/**
 * Author: ErosDiffusion (EF)
 * Email: erosdiffusionai+controlnetmaps@gmail.com
 * Year: 2025
 */

import { app } from "../../scripts/app.js";

// Simplified: always use the Lit implementation and remove preference switch.
app.registerExtension({
  name: "ErosDiffusion.CacheMapBrowser",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "CacheMapBrowserNode") {
      // Helper to load and create the lit drawer
      let drawerInstance = null;
      const getDrawer = async () => {
        if (drawerInstance) return drawerInstance;
        try {
          await import("./cache_map_browser_lit.js");
          const el = document.createElement("eros-lit-browser");
          // Prefer registering with Comfy's sidebar API when available
          try {
            const sidebar = app?.ui?.sidebar;
            if (sidebar) {
              // Try several possible sidebar registration method names for compatibility
              if (typeof sidebar.register === "function") {
                const id = sidebar.register("eros-cache-browser", {
                  title: "Cache Browser",
                  element: el,
                });
                el._sidebarMethod = "register";
                el._sidebarId = id || "eros-cache-browser";
                el._sidebar = sidebar;
                drawerInstance = el;
                return el;
              } else if (typeof sidebar.addTab === "function") {
                const id = sidebar.addTab("eros-cache-browser", el, { title: "Cache Browser" });
                el._sidebarMethod = "addTab";
                el._sidebarId = id || "eros-cache-browser";
                el._sidebar = sidebar;
                drawerInstance = el;
                return el;
              } else if (typeof sidebar.addPanel === "function") {
                const id = sidebar.addPanel("Cache Browser", el);
                el._sidebarMethod = "addPanel";
                el._sidebarId = id || "eros-cache-browser";
                el._sidebar = sidebar;
                drawerInstance = el;
                return el;
              }
            }
          } catch (ex) {
            console.warn("Sidebar registration failed, falling back to body append:", ex);
          }

          // Fallback: append to body and use the component's own open/close
          document.body.appendChild(el);
          drawerInstance = el;
          return el;
        } catch (e) {
          console.error("Failed to load CacheMap Browser (lit):", e);
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

          // If the component was registered with the sidebar, try to open/show it via sidebar APIs
          const sidebar = this.drawer._sidebar || app?.ui?.sidebar;
          if (sidebar) {
            try {
              const id = this.drawer._sidebarId || "eros-cache-browser";
              if (typeof sidebar.openTab === "function") {
                sidebar.openTab(id);
                return;
              } else if (typeof sidebar.open === "function") {
                sidebar.open(id);
                return;
              } else if (typeof sidebar.showTab === "function") {
                sidebar.showTab(id);
                return;
              } else if (typeof sidebar.activateTab === "function") {
                sidebar.activateTab(id);
                return;
              } else if (typeof sidebar.show === "function") {
                sidebar.show(id);
                return;
              }
            } catch (e) {
              console.warn("Sidebar open API failed, falling back to component.open:", e);
            }
          }

          // Fallback: call the component's open method
          if (typeof this.drawer.open === "function") this.drawer.open(this);
        });

        this.addWidget("button", "Run", "run", () => {
          app.queuePrompt(0, 1);
        });

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
