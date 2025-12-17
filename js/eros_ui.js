/**
 * Small UI helpers for ComfyUI extensions.
 * Wraps ComfyUI dialog/toast APIs with safe fallbacks.
 */

import { app } from "../../scripts/app.js";

export function toast({
  severity = "info",
  summary = "",
  detail = "",
  life = 3000,
} = {}) {
  try {
    const svc = app?.extensionManager?.toast;
    if (svc && typeof svc.add === "function") {
      svc.add({ severity, summary, detail, life });
      return;
    }
  } catch (e) {}

  // Fallback: avoid throwing; last-resort console.
  try {
    console.log("[ErosUI toast]", severity, summary, detail);
  } catch (e) {}
}

export async function confirmDialog({
  title = "Confirm",
  message = "Are you sure?",
  type = "default",
  itemList,
  hint,
} = {}) {
  try {
    const dlg = app?.extensionManager?.dialog;
    if (dlg && typeof dlg.confirm === "function") {
      const result = await dlg.confirm({ title, message, type, itemList, hint });
      return !!result;
    }
  } catch (e) {}

  // Fallback
  try {
    return window.confirm(message);
  } catch (e) {
    return false;
  }
}

export function downloadBlob(blob, filename) {
  try {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "download";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    // best-effort
  }
}
