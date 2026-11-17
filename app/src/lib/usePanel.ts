import { useEffect, useRef, useState, type FocusEvent, type ToggleEvent } from 'react'
import { placeAgainst, type PanelAlign } from './anchor'

type PanelOptions = {
  // the edge of the control the panel lines up with
  align?: PanelAlign
  // called when the panel has opened or closed
  onToggle?: (open: boolean) => void
}

// What every panel that opens from a control shares. It is the browser's own
// popover, so a click outside or Escape closes it and focus goes back to the
// control. It is placed before it is drawn, it stays against its control
// while the page scrolls under it, and it closes when the window changes
// size or its control leaves the window.
export function usePanel<Panel extends HTMLElement>({ align = 'start', onToggle }: PanelOptions = {}) {
  const control = useRef<HTMLButtonElement>(null)
  const panel = useRef<Panel>(null)
  const [open, setOpen] = useState(false)
  // how things stand after the latest render, for the moment the panel leaves the page
  const latest = useRef({ open, onToggle })

  useEffect(() => {
    latest.current = { open, onToggle }
  })

  // A panel taken off the page while it is open sends no event of its own, so
  // whoever asked to be told hears that it closed from here.
  useEffect(
    () => () => {
      if (latest.current.open) latest.current.onToggle?.(false)
    },
    [],
  )

  useEffect(() => {
    if (!open) return
    let frame = 0
    const close = () => panel.current?.hidePopover()
    const follow = (event: Event) => {
      // scrolling inside the panel is the panel's own business
      if (event.target instanceof Node && panel.current?.contains(event.target)) return
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        if (!control.current || !panel.current) return
        const at = control.current.getBoundingClientRect()
        if (at.bottom < 0 || at.top > window.innerHeight) close()
        else placeAgainst(control.current, panel.current, align)
      })
    }
    window.addEventListener('resize', close)
    document.addEventListener('scroll', follow, true)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', close)
      document.removeEventListener('scroll', follow, true)
    }
  }, [open, align])

  return {
    control,
    panel,
    open,
    close: () => panel.current?.hidePopover(),
    // spread onto the panel element, along with its id and popover="auto"
    panelEvents: {
      onBeforeToggle(event: ToggleEvent<Panel>) {
        if (event.newState === 'open' && control.current && panel.current) {
          placeAgainst(control.current, panel.current, align)
        }
      },
      onToggle(event: ToggleEvent<Panel>) {
        const opening = event.newState === 'open'
        setOpen(opening)
        onToggle?.(opening)
      },
      // The keyboard went on to something else on the page: the panel does
      // not stay open behind it. Back to its own control is not leaving.
      onBlur(event: FocusEvent<Panel>) {
        const to = event.relatedTarget
        if (to instanceof Node && to !== control.current && !panel.current?.contains(to)) panel.current?.hidePopover()
      },
    },
  }
}
