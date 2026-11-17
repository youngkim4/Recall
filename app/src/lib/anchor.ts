// Where a small panel opens relative to the control that owns it.

const GAP = 4
const EDGE = 8
// with less room than this under the control, the panel opens above it
const ROOM_WANTED = 280

// which edge of the control the panel lines up with
export type PanelAlign = 'start' | 'end'

// Places a fixed panel against its control: under it when there is room,
// above it otherwise, lined up with its left or right edge, and never past
// the window's edge. The panel grows from the corner that touches the
// control, so it reads as coming out of it. Called before the panel is shown,
// so nothing is ever drawn in the wrong place.
export function placeAgainst(control: HTMLElement, panel: HTMLElement, align: PanelAlign = 'start') {
  const anchor = control.getBoundingClientRect()
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
  const width = parseFloat(getComputedStyle(panel).width) || 18 * rem
  const below = window.innerHeight - anchor.bottom - GAP - EDGE
  const above = anchor.top - GAP - EDGE
  const opensUp = below < ROOM_WANTED && above > below
  const wanted = align === 'end' ? anchor.right - width : anchor.left
  const left = Math.max(EDGE, Math.min(wanted, window.innerWidth - width - EDGE))
  // the corner it grows from is the one nearest the control's own edge
  const fromRight = Math.abs(left + width - anchor.right) < Math.abs(left - anchor.left)

  panel.style.left = `${left}px`
  panel.style.top = opensUp ? 'auto' : `${anchor.bottom + GAP}px`
  panel.style.bottom = opensUp ? `${window.innerHeight - anchor.top + GAP}px` : 'auto'
  panel.style.maxHeight = `${Math.max(160, opensUp ? above : below)}px`
  panel.style.setProperty('--origin', `${opensUp ? 'bottom' : 'top'} ${fromRight ? 'right' : 'left'}`)
}
