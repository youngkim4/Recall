// room kept over something that is brought to the top of what scrolls
const HEADROOM_PX = 24

// Brings part of a report into view by moving only what scrolls the report:
// its pane in the app, the window on paper. scrollIntoView would also shift
// every clipped box around the pane, the app's own frame among them.
export function bringIntoView(element: HTMLElement, place: 'start' | 'center') {
  const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
  const box = element.getBoundingClientRect()
  const pane = document.documentElement.hasAttribute('data-paper') ? null : element.closest<HTMLElement>('.pane-detail')
  const frame = pane ? pane.getBoundingClientRect() : { top: 0, height: window.innerHeight }
  const room = place === 'center' ? Math.max(HEADROOM_PX, (frame.height - box.height) / 2) : HEADROOM_PX
  const by = box.top - frame.top - room
  if (pane) pane.scrollTo({ top: pane.scrollTop + by, behavior })
  else window.scrollTo({ top: window.scrollY + by, behavior })
}
