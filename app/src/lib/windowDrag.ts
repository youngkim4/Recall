// The Mac window has no title bar of its own to take hold of: the page
// covers it. So the page says when a press is on the bare top of the
// window, and the shell moves the window with it; a double click there
// zooms the window, as on any title bar.
import { recallBridge } from './bridge.ts'

// as tall as the title bar the window would have
export const DRAG_ZONE = 28

// What takes a press for itself, so a press on it does not move the window:
// the controls, words that can be selected, and anything marked data-no-drag.
const PRESSED = [
  'button, a, input, select, textarea, label, summary, [role], [tabindex], [contenteditable], .resize-handle',
  'p, h1, h2, h3, h4, dt, dd, li, blockquote, span, .msg, .bubble, [data-no-drag]',
].join(', ')

type Press = { button: number; clientY: number; detail: number; target: unknown }
type Target = { closest?: (selector: string) => unknown }

// what a press at the top of the window does: nothing, move the window, or zoom it
export function pressOf(press: Press): 'none' | 'drag' | 'zoom' {
  if (press.button !== 0 || press.clientY >= DRAG_ZONE) return 'none'
  const target = press.target as Target | null
  if (!target?.closest || target.closest(PRESSED)) return 'none'
  return press.detail === 2 ? 'zoom' : 'drag'
}

export function moveWindowFromItsTop(): () => void {
  function onPress(event: MouseEvent) {
    const what = pressOf(event)
    if (what === 'none') return
    // The press moves the window and selects no text. A press on bare
    // ground still takes the keyboard from a field, as any press there does.
    event.preventDefault()
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    if (what === 'zoom') recallBridge.zoomWindow()
    else recallBridge.dragWindow()
  }
  document.addEventListener('mousedown', onPress)
  return () => document.removeEventListener('mousedown', onPress)
}
