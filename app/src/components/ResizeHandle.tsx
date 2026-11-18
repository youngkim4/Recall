import { useEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react'

type ResizeHandleProps = {
  label: string
  value: number
  min: number
  max: number
  onChange: (value: number) => void
  // true while there is nothing to resize, as with a folded rail
  disabled?: boolean
  // the pane this edge sizes is on its right: it grows as the edge moves left
  sizesRight?: boolean
}

// how far one press of an arrow key moves the edge
const KEY_STEP = 16

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

// The edge between two panes. Drag it, or focus it and use the arrow keys;
// Home and End take it to its narrowest and widest.
export function ResizeHandle({ label, value, min, max, onChange, disabled = false, sizesRight = false }: ResizeHandleProps) {
  // which way a move to the right takes the width
  const way = sizesRight ? -1 : 1
  // where a drag began: the pointer's place, and the width at that moment
  const drag = useRef<{ x: number; value: number } | null>(null)

  // a handle taken away in the middle of a drag leaves nothing behind
  useEffect(() => () => document.body.classList.remove('is-resizing'), [])

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (disabled || event.button !== 0) return
    event.preventDefault()
    // the handle keeps the pointer, so the drag ends here wherever it is let go
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = { x: event.clientX, value }
    document.body.classList.add('is-resizing')
    event.currentTarget.classList.add('is-dragging')
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    const from = drag.current
    if (from) onChange(clamp(from.value + (event.clientX - from.x) * way, min, max))
  }

  function endDrag(event: PointerEvent<HTMLDivElement>) {
    if (!drag.current) return
    drag.current = null
    document.body.classList.remove('is-resizing')
    event.currentTarget.classList.remove('is-dragging')
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const to =
      event.key === 'ArrowLeft'
        ? value - KEY_STEP * way
        : event.key === 'ArrowRight'
          ? value + KEY_STEP * way
          : event.key === 'Home'
            ? min
            : event.key === 'End'
              ? max
              : null
    if (to === null || disabled) return
    event.preventDefault()
    onChange(clamp(to, min, max))
  }

  return (
    <div
      className="resize-handle"
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-disabled={disabled || undefined}
      tabIndex={disabled ? -1 : 0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onLostPointerCapture={endDrag}
      onKeyDown={handleKeyDown}
    />
  )
}
