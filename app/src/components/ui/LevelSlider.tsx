import { useId, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react'
import { alongAt, stepAt, stepByKey } from '../../lib/levelSlider'

export type LevelStep = { value: string; label: string }

type LevelSliderProps = {
  // what is being set: "Reasoning"
  label: string
  // the values it can take, least first
  steps: LevelStep[]
  value: string
  onChange: (value: string) => void
}

// Half the height of the track, in px: how far in from each end the first
// and last steps stand. The style sheet holds the height as --track.
const INSET = 15

// A slider with a few named steps: a thick track with a mark at each step,
// filled up to a round knob. Its name and the step it is on stand over it.
// A press brings the knob to the pointer, and the knob then follows the
// pointer; let go, it settles on the nearest step. The step changes as the
// knob passes the middle between two marks. The arrow keys move one step.
export function LevelSlider({ label, steps, value, onChange }: LevelSliderProps) {
  const labelId = useId()
  // how far along the knob is held while it is dragged, 0 to 1; null when it rests on its step
  const [held, setHeld] = useState<number | null>(null)
  // the one pointer that holds the knob; a second finger on the track is not followed
  const holder = useRef<number | null>(null)
  const at = Math.max(
    0,
    steps.findIndex((step) => step.value === value),
  )
  const last = steps.length - 1

  function go(index: number) {
    const step = steps[index]
    if (step && step.value !== value) onChange(step.value)
  }

  function follow(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    setHeld(alongAt(event.clientX, box, INSET))
    go(stepAt(event.clientX, box, INSET, steps.length))
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || holder.current !== null) return
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // A pointer that can not be held sets the step it pressed and does not
      // drag: with no hold, nothing would say when it lets go.
      const box = event.currentTarget.getBoundingClientRect()
      go(stepAt(event.clientX, box, INSET, steps.length))
      return
    }
    holder.current = event.pointerId
    follow(event)
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerId === holder.current && event.currentTarget.hasPointerCapture(event.pointerId)) follow(event)
  }

  function handlePointerEnd(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerId === holder.current) release()
  }

  // The knob goes back to its step. This also runs when the keyboard leaves
  // the track, which is what happens when the panel around it closes in the
  // middle of a drag: no pointer event may come after that.
  function release() {
    holder.current = null
    setHeld(null)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    const next = stepByKey(event.key, at, steps.length)
    if (next === null) return
    event.preventDefault()
    go(next)
  }

  if (steps.length < 2) return null
  const along = held ?? at / last
  return (
    <div className="level-slider">
      <div className="level-slider-head">
        <span id={labelId}>{label}</span>
        {/* a new step fades in over the old one */}
        <span key={value} className="level-slider-value swap" aria-hidden="true">
          {steps[at].label}
        </span>
      </div>
      <div
        className="level-slider-track"
        role="slider"
        tabIndex={0}
        aria-labelledby={labelId}
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={at}
        aria-valuetext={steps[at].label}
        data-held={held !== null || undefined}
        style={{ '--at': along } as CSSProperties}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        onLostPointerCapture={handlePointerEnd}
        onBlur={release}
        onKeyDown={handleKeyDown}
      >
        <span className="level-slider-fill" />
        {steps.map((step, index) => (
          <span
            key={step.value}
            className="level-slider-stop"
            // a mark the fill has reached stands on the fill
            data-passed={index / last <= along || undefined}
            style={{ '--stop': index / last } as CSSProperties}
          />
        ))}
        <span className="level-slider-knob" />
      </div>
    </div>
  )
}
