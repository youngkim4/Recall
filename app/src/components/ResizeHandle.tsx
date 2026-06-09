import type { PointerEvent as ReactPointerEvent } from 'react'

type ResizeHandleProps = {
  label: string
  value: number
  min: number
  max: number
  onChange: (value: number) => void
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

export function ResizeHandle({ label, value, min, max, onChange }: ResizeHandleProps) {
  function handlePointerDown(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault()
    const startX = event.clientX
    const startValue = value
    const handle = event.currentTarget

    document.body.classList.add('is-resizing')
    handle.classList.add('is-dragging')

    function handlePointerMove(pointerEvent: PointerEvent) {
      onChange(clamp(startValue + pointerEvent.clientX - startX, min, max))
    }

    function handlePointerUp() {
      document.body.classList.remove('is-resizing')
      handle.classList.remove('is-dragging')
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
  }

  return (
    <button
      type="button"
      className="resize-handle"
      aria-label={label}
      onPointerDown={handlePointerDown}
    />
  )
}
