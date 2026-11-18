import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
} from 'react'
import { recallBridge } from '../../lib/bridge'
import { alongAt, heldStep, stepAt, stepByKey } from '../../lib/levelSlider'

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
const INSET = 11
// A press away from the knob brings the knob to the pointer: each frame it
// covers this share of the way that is left, and it has arrived when less
// than half a pixel is left.
const CHASE = 0.32
const ARRIVED_PX = 0.5
// How long one pulse of the track takes on the first step and on the last,
// in seconds: the more the model thinks, the faster the track beats. Even
// the last step is slower than a resting heart.
const PULSE_SLOW = 4.2
const PULSE_FAST = 2.4
const SWELL = 'cubic-bezier(0.34, 1.56, 0.64, 1)'
const OUT = 'cubic-bezier(0.23, 1, 0.32, 1)'

function still() {
  return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

// A slider with a few named steps: a track with a mark at each step, filled
// up to a round knob. Its name and the step it is on stand over it.
//
// The knob is moved by the pointer itself: it stays under the pointer where
// it was taken, and the page is not drawn again for each move. A press away
// from the knob first brings the knob there, with no jump at the end, also
// when the pointer moves on in the meantime. The step changes when the knob
// is well past half way to the next one, so a hand that rests between two
// steps does not flip between them. On a new step the knob and the step's
// mark swell once, a ring goes out from the knob, and the trackpad ticks;
// the last step also sends a light along the track. Let go, the knob
// springs onto its step. The arrow keys move one step.
export function LevelSlider({ label, steps, value, onChange }: LevelSliderProps) {
  const labelId = useId()
  const root = useRef<HTMLDivElement>(null)
  const track = useRef<HTMLDivElement>(null)
  // The one pointer that holds the knob: where the track was when it
  // pressed, how far the knob's middle was from it then (as a share of the
  // track), where it wants the knob now and where the knob is shown. While
  // the knob is still on its way to the pointer, `frame` is the next step of
  // that way.
  const holder = useRef<{
    id: number
    span: { left: number; width: number }
    offset: number
    target: number
    shown: number
    frame: number
  } | null>(null)
  // the step last chosen here, for the moment the knob is let go
  const chosen = useRef(0)
  // which way the name of the step last turned
  const [turn, setTurn] = useState<'up' | 'down'>('up')
  const at = Math.max(
    0,
    steps.findIndex((step) => step.value === value),
  )
  const last = steps.length - 1

  // where the knob is
  function place(along: number) {
    root.current?.style.setProperty('--at', String(along))
  }

  // How fast the track beats on a step. It is set when the step changes and
  // not while the knob moves: a beat whose length changed at each move would
  // start again at each move.
  function pace(index: number) {
    const seconds = PULSE_SLOW - (PULSE_SLOW - PULSE_FAST) * (index / Math.max(1, last))
    root.current?.style.setProperty('--pulse', `${seconds.toFixed(2)}s`)
    root.current?.style.setProperty('--level', String(index / Math.max(1, last)))
  }

  // The knob stands on its step, unless a pointer holds it: then the pointer
  // says where it is.
  useLayoutEffect(() => {
    chosen.current = at
    pace(at)
    if (holder.current || last < 1) return
    place(at / last)
  }, [at, last])

  // a slider that leaves the page in the middle of a drag lets go of its pointer
  useEffect(
    () => () => {
      if (holder.current) cancelAnimationFrame(holder.current.frame)
      holder.current = null
    },
    [],
  )

  function beat(index: number) {
    recallBridge.tick()
    const box = track.current
    if (!box || still()) return
    // the knob is a little larger while it is held
    const base = box.hasAttribute('data-held') ? 1.06 : 1
    box
      .querySelector('.level-slider-knob')
      ?.animate([{ scale: base }, { scale: base * 1.2 }, { scale: base }], { duration: 300, easing: SWELL })
    box
      .querySelector('.level-slider-ring')
      ?.animate([{ opacity: 0.6, scale: 1 }, { opacity: 0, scale: 2.8 }], { duration: 620, easing: OUT })
    for (const mark of box.querySelectorAll(`[data-step="${index}"]`)) {
      mark.animate([{ scale: 1 }, { scale: 2.6 }, { scale: 1 }], { duration: 300, easing: SWELL })
    }
    if (index === last) {
      box.querySelector('.level-slider-shine')?.animate(
        [
          { opacity: 0, translate: '-120% 0' },
          { opacity: 0.55, offset: 0.3 },
          { opacity: 0, translate: '420% 0' },
        ],
        { duration: 720, easing: 'ease-out' },
      )
    }
  }

  function go(index: number) {
    const step = steps[index]
    if (!step || index === chosen.current) return
    setTurn(index > chosen.current ? 'up' : 'down')
    chosen.current = index
    pace(index)
    onChange(step.value)
    beat(index)
  }

  // One step of the knob's way to the pointer. It ends on the pointer
  // exactly, and from then on the pointer moves the knob itself.
  function chase() {
    const held = holder.current
    if (!held) return
    const left = held.target - held.shown
    const arrived = Math.abs(left) * (held.span.width - INSET * 2) < ARRIVED_PX
    held.shown = arrived ? held.target : held.shown + left * CHASE
    held.frame = arrived ? 0 : requestAnimationFrame(chase)
    place(held.shown)
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || holder.current) return
    const box = event.currentTarget
    const rect = box.getBoundingClientRect()
    const span = { left: rect.left, width: rect.width }
    try {
      box.setPointerCapture(event.pointerId)
    } catch {
      // A pointer that can not be held sets the step it pressed and does not
      // drag: with no hold, nothing would say when it lets go.
      go(stepAt(event.clientX, span, INSET, steps.length))
      return
    }
    const pointer = alongAt(event.clientX, span, INSET)
    const knob = chosen.current / last
    // A press on the knob takes it where it is: the knob does not jump to
    // the middle of the pointer. A press elsewhere brings the knob there,
    // and the step is the one that was pressed.
    const onKnob = Math.abs(knob - pointer) * (span.width - INSET * 2) <= INSET
    const offset = onKnob ? knob - pointer : 0
    holder.current = { id: event.pointerId, span, offset, target: pointer + offset, shown: knob, frame: 0 }
    box.setAttribute('data-held', '')
    if (onKnob) return
    go(stepAt(event.clientX, span, INSET, steps.length))
    holder.current.frame = requestAnimationFrame(chase)
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    const held = holder.current
    if (!held || event.pointerId !== held.id) return
    held.target = Math.min(1, Math.max(0, alongAt(event.clientX, held.span, INSET) + held.offset))
    // a knob that has come to the pointer is moved by it; one still on its way keeps chasing
    if (!held.frame) {
      held.shown = held.target
      place(held.shown)
    }
    go(heldStep(held.target, steps.length, chosen.current))
  }

  function handlePointerEnd(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerId === holder.current?.id) release()
  }

  // The knob springs onto its step. This also runs when the keyboard leaves
  // the track, which is what happens when the panel around it closes in the
  // middle of a drag: no pointer event may come after that.
  function release() {
    if (holder.current) cancelAnimationFrame(holder.current.frame)
    holder.current = null
    track.current?.removeAttribute('data-held')
    if (last > 0) place(chosen.current / last)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    const next = stepByKey(event.key, at, steps.length)
    if (next === null) return
    event.preventDefault()
    go(next)
  }

  if (steps.length < 2) return null
  return (
    <div ref={root} className="level-slider">
      <div className="level-slider-head">
        <span id={labelId}>{label}</span>
        {/* the name of a new step turns in from the side the knob came from */}
        <span key={value} className="level-slider-value" data-turn={turn} aria-hidden="true">
          {steps[at].label}
        </span>
      </div>
      <div
        ref={track}
        className="level-slider-track"
        role="slider"
        tabIndex={0}
        aria-labelledby={labelId}
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={at}
        aria-valuetext={steps[at].label}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        onLostPointerCapture={handlePointerEnd}
        onBlur={release}
        onKeyDown={handleKeyDown}
      >
        {/* the light the fill throws around the track: it beats, faster on a higher step */}
        <span className="level-slider-aura" />
        {steps.map((step, index) => (
          <span
            key={step.value}
            className="level-slider-stop"
            data-step={index}
            style={{ '--stop': index / last } as CSSProperties}
          />
        ))}
        {/* the fill carries its own marks, so a mark is pale exactly where the fill has reached it */}
        <span className="level-slider-fill">
          {steps.map((step, index) => (
            <span
              key={step.value}
              className="level-slider-stop"
              data-step={index}
              style={{ '--stop': index / last } as CSSProperties}
            />
          ))}
          <span className="level-slider-shine" />
        </span>
        <span className="level-slider-ring" />
        <span className="level-slider-knob" />
      </div>
    </div>
  )
}
