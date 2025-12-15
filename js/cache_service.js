import { api } from "../../scripts/api.js";

// ========================================================
// Service Layer: Pure Logic & API Encapsulation
// ========================================================
export class CacheService {
    constructor() {
        this.allTags = new Map();
        this.imageTags = new Map(); // basename -> Set<tag>
        this.cachePath = "";
        this.listeners = new Set();
    }

    setCachePath(path) { this.cachePath = path; }

    subscribe(cb) { this.listeners.add(cb); }
    unsubscribe(cb) { this.listeners.delete(cb); }
    notify(event, data) { this.listeners.forEach(cb => cb(event, data)); }

    getBasename(filename) {
        return filename.split('/').pop().replace(/\.[^/.]+$/, "");
    }

    async loadTags() {
        try {
            const resp = await api.fetchApi("/eros/tags/list");
            const data = await resp.json();
            this.allTags.clear();
            if (data.tags) data.tags.forEach(t => this.allTags.set(t.name, t.count));
            this.notify("tags-loaded", this.allTags);
        } catch (e) { console.error("API Error:", e); }
    }

    async fetchFiles(subfolder) {
        try {
            const url = `/eros/cache/fetch_files?path=${encodeURIComponent(this.cachePath)}&subfolder=${encodeURIComponent(subfolder)}`;
            const resp = await api.fetchApi(url);
            const data = await resp.json();
            return data.files || [];
        } catch (e) {
            console.error("API Error:", e);
            return [];
        }
    }

    async loadImageTags(basename) {
        if (!basename) return new Set();
        try {
            const resp = await api.fetchApi("/eros/tags/for_image?path=" + encodeURIComponent(basename));
            const data = await resp.json();
            const tags = new Set(data.tags || []);
            this.imageTags.set(basename, tags);
            this.notify("tag-added", { basename }); // Reuse tag-added to trigger re-renders
            return tags;
        } catch { return new Set(); }
    }

    async addTag(basename, tag) {
        if (!basename) return;
        // Optimistic
        if (!this.imageTags.has(basename)) this.imageTags.set(basename, new Set());
        this.imageTags.get(basename).add(tag);
        this.notify("tag-added", { basename, tag });

        try {
            await api.fetchApi("/eros/tags/add_to_image", {
                method: "POST", body: JSON.stringify({ path: basename, tag: tag })
            });
        } catch (e) {
            console.error("Add Tag Failed:", e);
        }
    }

    async removeTag(basename, tag) {
        if (!basename) return;
        // Optimistic
        if (this.imageTags.has(basename)) {
            this.imageTags.get(basename).delete(tag);
            this.notify("tag-removed", { basename, tag });
        }
        try {
            await api.fetchApi("/eros/tags/remove_from_image", {
                method: "POST", body: JSON.stringify({ path: basename, tag: tag })
            });
        } catch (e) {
            console.error("Remove Tag Failed:", e);
        }
    }

    async autoTag(basename) {
        if (!basename) return null;
        try {
            const resp = await api.fetchApi("/eros/tags/auto_tag", {
                method: "POST", body: JSON.stringify({ path: basename })
            });
            return await resp.json();
        } catch (e) { return null; }
    }
}
