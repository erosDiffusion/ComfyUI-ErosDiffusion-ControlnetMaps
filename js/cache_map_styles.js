/**
 * Author: ErosDiffusion (EF)
 * Email: erosdiffusionai+controlnetmaps@gmail.com
 * Year: 2025
 */

const STYLE_ID = "eros-cache-browser-style";
export const DRAWER_CSS = `
.eros-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: 960px; /* 2x larger default */
    height: 100vh;
    background: rgba(20, 20, 24, 0.98);
    backdrop-filter: blur(10px);
    border-left: 1px solid #333;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    padding: 0; /* Remove padding here, managing in children */
    box-shadow: -5px 0 15px rgba(0,0,0,0.5);
    color: white;
    font-family: sans-serif;
    transform: translateX(100%);
    transition: transform 0.3s ease-in-out;
    min-width: 400px;
}
.eros-drawer.open {
    transform: translateX(0);
}
.eros-drawer-resize-handle {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 6px;
    cursor: ew-resize;
    z-index: 10000;
}
.eros-drawer-resize-handle:hover {
    background: rgba(255, 255, 255, 0.1);
}

.eros-drawer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #444;
    padding: 8px 15px;
    height: 40px;
    flex-shrink: 0;
}
.eros-drawer-header h3 { margin: 0; font-size: 14px; }
.eros-drawer-close {
    cursor: pointer;
    font-size: 24px;
    color: #888;
    transition: color 0.2s;
}
.eros-drawer-close:hover { color: white; }

/* Main Content Wrapper (2 Columns) */
.eros-drawer-content-wrapper {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* Column 1: Main Controls & Grid */
.eros-main-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 20px;
    overflow: hidden;
}

/* Column 2: Tag Sidebar */
.eros-tag-sidebar {
    width: 300px; /* Fixed width for tags, or could be flex */
    min-width: 250px;
    border-left: 1px solid #444;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
    background: rgba(0, 0, 0, 0.2);
    overflow-y: auto;
}

.eros-controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 15px;
    flex-shrink: 0;
}

/* Tabs */
.eros-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 5px;
}
.eros-tab {
    padding: 1px 6px;
    background: #2a2a2a;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    border: 1px solid #3d3d3d;
    transition: all 0.2s;
    user-select: none;
    text-transform: capitalize;
    line-height: 10px;
}
.eros-tab:hover {
    background: #3a3a3a;
    border-color: #555;
    color: #eee;
}
.eros-tab.active {
    background: var(--color-interactive-active, #2196F3);
    border-color: var(--color-interactive-active, #2196F3);
    color: white;
    font-weight: bold;
}

/* Utils Bar - Hidden as requested */
.eros-utils {
    display: flex;
    gap: 10px;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    display: none;
}

/* 2x3 Grid for Controls */
.eros-control-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: auto auto;
    gap: 8px;
    align-items: center;
}

.eros-path-select {
    padding: 4px;
    background: #222;
    color: #ddd;
    border: 1px solid #444;
    border-radius: 4px;
    font-size: 11px;
    grid-column: span 3; /* Path select full width if needed, or adjust */
}

.eros-btn {
    padding: 4px 8px;
    background: #333;
    color: white;
    border: 1px solid #555;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    transition: background 0.2s;
    text-align: center;
}
.eros-btn:hover { background: #444; }

.eros-overlay-toggle {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #ccc;
    cursor: pointer;
}
.eros-overlay-controls {
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 10px;
    color: #ccc;
}
.eros-overlay-controls label {
    display: flex;
    align-items: center;
    gap: 5px;
}
.eros-overlay-controls input[type=range],
.eros-overlay-controls select {
    width: 100%;
    font-size: 10px;
    background: #222;
    color: #ddd;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 2px 4px;
}
.eros-overlay-controls input[type=range] {
    padding: 0;
}

/* Grid Images */
.eros-grid {
    --grid-columns: 4;
    display: grid;
    grid-template-columns: repeat(var(--grid-columns), 1fr);
    grid-auto-rows: min-content; /* Prevent row stretching */
    gap: 10px;
    padding-right: 5px; /* Scrollbar space */
    overflow-y: auto;
    flex: 1;
    align-content: start;
}
.eros-item {
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    background: #1a1a1a;
    border-radius: 4px;
    cursor: pointer;
    transition: box-shadow 0.2s; /* Removed transform transition */
    flex: none;
}
.eros-item:hover {
    /* Removed transform: scale(1.02); */
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.eros-item.selected {
    outline: 3px solid var(--color-interactive-active, #2196F3);
    outline-offset: -3px;
}
.eros-item img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}
.eros-overlay {
    opacity: 0.2;
    pointer-events: none;
    z-index: 2;
}
.eros-item-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.8));
    color: white;
    font-size: 10px;
    padding: 4px;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    pointer-events: none;
    z-index: 2;
}
/* Star */
.eros-star {
    position: absolute;
    top: 4px;
    right: 4px;
    font-size: 18px;
    color: #fff;
    opacity: 0.6;
    cursor: pointer;
    z-index: 5;
    text-shadow: 0 0 3px rgba(0,0,0,0.8);
    transition: opacity 0.2s, transform 0.2s;
    user-select: none;
}
.eros-star:hover {
    opacity: 1;
    transform: scale(1.2);
}
.eros-star.active {
    color: #FFD700;
    opacity: 1;
}

/* Tag Badges on Image */
.eros-tag-badges {
    position: absolute;
    bottom: 22px;
    left: 2px;
    right: 2px;
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    z-index: 4;
    pointer-events: none;
    max-height: 40%;
    overflow: hidden;
}
.eros-tag-badge {
    background: rgba(33, 150, 243, 0.9);
    color: white;
    font-size: var(--badge-font-size, 9px);
    padding: 1px 4px;
    border-radius: 2px;
    max-width: 70px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Tag Sidebar Section */
.eros-tag-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 20px;
}
.eros-tag-section-label {
    font-size: 11px;
    color: #aaa;
    font-weight: bold;
    text-transform: uppercase;
    border-bottom: 1px solid #444;
    padding-bottom: 4px;
    margin-bottom: 4px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
}
.eros-tag-section-label:hover {
    color: white;
}
.eros-tag-section-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.eros-tag-section-content.collapsed {
    display: none;
}

.eros-tag-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
}
.eros-tag-chip {
    background: #2a2a2a;
    color: #9da2ad;
    font-size: 10px;
    padding: 0px 2px;
    border-radius: 4px;
    /* border: 1px solid #cfcfcf; */
    user-select: none;
    display: flex;
    align-items: center;
    gap: 0px;
    transition: all 0.2s;
    font-family: monospace;
    line-height: 10px;
}
.eros-tag-chip:hover {
    background: #3a3a3a;
    color: #fff;
}
.eros-tag-chip.active {
    background: var(--color-interactive-active, #2196F3);
    color: white;
}
.eros-tag-chip-remove {
    font-size: 12px;
    margin-left: 4px;
    opacity: 0.7;
    cursor: pointer;
}
.eros-tag-chip-remove:hover { opacity: 1; }

.eros-tag-input-row {
    display: flex;
    gap: 4px;
}
.eros-tag-input {
    flex: 1;
    background: #222;
    color: #ddd;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 10px;
}
.eros-tag-add-btn {
    background: #333;
    color: white;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px 8px;
    font-size: 10px;
    cursor: pointer;
}
.eros-tag-add-btn:hover { background: #444; }
`;

export function injectStyles() {
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.innerText = DRAWER_CSS;
    document.head.appendChild(style);
  }
}
