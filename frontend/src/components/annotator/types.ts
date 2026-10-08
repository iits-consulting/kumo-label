import type { Snippet } from "svelte";

/**
 * Render fragments DetectionPanel hands back to AnnotatorOverlay.
 *
 * The React version was a JSX-returning hook (`useDetectionPanel`) whose six
 * render pieces were placed at different spots of the overlay layout. In
 * Svelte, DetectionPanel is a real component: the overlay defines its whole
 * layout as a `layout(det)` snippet and, for detection, renders it *inside*
 * DetectionPanel's `children` snippet so the panel can pass these fragments up.
 */
export interface DetectionFragments {
  canvasHandlers: {
    onMouseDown: (e: MouseEvent) => void;
    onMouseMove: (e: MouseEvent) => void;
    onMouseUp: (e: MouseEvent) => void;
    onMouseLeave: (e: MouseEvent) => void;
  };
  toolbar: Snippet;
  canvasOverlay: Snippet;
  panel: Snippet;
  stats: Snippet;
  helpText: string;
}
