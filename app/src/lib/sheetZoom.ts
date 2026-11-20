// A sheet that grows out of what was clicked to open it, and goes back into
// it when it closes: the card lifts off the page, widens into the sheet, and
// its words give way to the sheet's own. The origin is hidden while the sheet
// is open, so the sheet is the card and not a copy of it. With reduced motion,
// or with no origin on the page, the sheet only fades (cards.css).
import { zoomFrame, zoomKeyframes, type ZoomFrame } from './zoom.ts'

// open: long enough to follow, decelerating into place (Material's emphasized curve)
const OPEN_MS = 460
const OPEN_EASE = 'cubic-bezier(0.2, 0, 0, 1)'
// close: about half as long, and moving at once. A curve that starts slow
// (an accelerating one) leaves the sheet standing while the reader waits.
const CLOSE_MS = 240
const CLOSE_EASE = OPEN_EASE
// The shares of the way over which the card's words give way to the sheet's:
// opening, the card's go first and the sheet's follow; closing, the other way.
const OPEN_CARD_OUT = [0, 0.3] as const
const OPEN_SHEET_IN = [0.15, 0.6] as const
const CLOSE_SHEET_OUT = [0, 0.25] as const
const CLOSE_CARD_IN = [0.15, 0.55] as const

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
const radiusOf = (element: Element) => parseFloat(getComputedStyle(element).borderTopLeftRadius) || 0

function setHidden(element: Element, hidden: boolean) {
  if (element instanceof HTMLElement) element.style.visibility = hidden ? 'hidden' : ''
}

function frameOf(sheet: HTMLElement, origin: Element | null | undefined): ZoomFrame | null {
  if (!origin?.isConnected || reducedMotion()) return null
  return zoomFrame(origin.getBoundingClientRect(), sheet.getBoundingClientRect())
}

// The card as it looks, drawn inside the sheet at the card's size, so it
// grows with the sheet while it fades. Nothing in it can be read or clicked.
function standIn(origin: Element, frame: ZoomFrame): HTMLElement {
  const copy = origin.cloneNode(true) as HTMLElement
  for (const element of [copy, ...copy.querySelectorAll('[id]')]) element.removeAttribute('id')
  copy.setAttribute('aria-hidden', 'true')
  copy.inert = true
  const { width, height } = origin.getBoundingClientRect()
  Object.assign(copy.style, {
    position: 'absolute',
    inset: '0 auto auto 0',
    width: `${width}px`,
    height: `${height}px`,
    margin: '0',
    boxSizing: 'border-box',
    boxShadow: 'none',
    visibility: 'visible',
    pointerEvents: 'none',
    transformOrigin: '0 0',
    transform: `scale(${1 / frame.scale})`,
  })
  return copy
}

// from one opacity to another between two shares of the way, held before and after
const fade = (from: number, to: number, [start, end]: readonly [number, number]): Keyframe[] => [
  { opacity: from, offset: 0 },
  { opacity: from, offset: start },
  { opacity: to, offset: end },
  { opacity: to, offset: 1 },
]

// Grows the sheet, just opened, out of `origin`. Returns what undoes it:
// called when the sheet goes away, it ends the motion and shows the origin.
export function zoomIn(sheet: HTMLElement, origin: Element | null | undefined): () => void {
  const frame = frameOf(sheet, origin)
  if (!frame || !origin) return () => {}
  const [start, end] = zoomKeyframes(frame, radiusOf(origin), radiusOf(sheet))
  const words = [...sheet.children]
  const copy = standIn(origin, frame)
  sheet.append(copy)
  setHidden(origin, true)
  const timing = { duration: OPEN_MS, easing: OPEN_EASE }
  const motions = [
    sheet.animate([start, end], timing),
    copy.animate(fade(1, 0, OPEN_CARD_OUT), { duration: OPEN_MS }),
    ...words.map((element) => element.animate(fade(0, 1, OPEN_SHEET_IN), { duration: OPEN_MS })),
  ]
  motions[1].finished.then(
    () => copy.remove(),
    () => copy.remove(),
  )
  return () => {
    for (const motion of motions) motion.cancel()
    copy.remove()
    setHidden(origin, false)
  }
}

// Shrinks the sheet back into `origin`, or fades it where the origin is gone.
// Resolves when the sheet can be closed; the origin is shown again by then.
export async function zoomOut(sheet: HTMLElement, origin: Element | null | undefined): Promise<void> {
  for (const motion of sheet.getAnimations({ subtree: true })) motion.finish()
  if (reducedMotion()) return
  const frame = frameOf(sheet, origin)
  if (!frame || !origin) {
    const timing: KeyframeAnimationOptions = { duration: CLOSE_MS / 2, fill: 'forwards' }
    await within(
      CLOSE_MS / 2,
      settle(sheet.animate([{ opacity: 1 }, { opacity: 0 }], timing)),
      backdropFade(sheet, timing),
    )
    return
  }
  const backdrop = backdropFade(sheet, { duration: CLOSE_MS, fill: 'forwards' })
  const [start, end] = zoomKeyframes(frame, radiusOf(origin), radiusOf(sheet))
  const copy = standIn(origin, frame)
  const words = [...sheet.children]
  sheet.append(copy)
  const timing: KeyframeAnimationOptions = { duration: CLOSE_MS, easing: CLOSE_EASE, fill: 'forwards' }
  copy.animate(fade(0, 1, CLOSE_CARD_IN), { duration: CLOSE_MS, fill: 'forwards' })
  for (const element of words) element.animate(fade(1, 0, CLOSE_SHEET_OUT), { duration: CLOSE_MS, fill: 'forwards' })
  await within(CLOSE_MS, settle(sheet.animate([end, start], timing)), backdrop)
  // the copy now stands where the card is: the card comes back under it before the sheet goes
  setHidden(origin, false)
}

// The dim behind the sheet fades with it. A browser that cannot move a
// pseudo-element skips it; the dim then goes with the sheet.
function backdropFade(sheet: HTMLElement, timing: KeyframeAnimationOptions): Promise<void> {
  try {
    return settle(sheet.animate([{ opacity: 1 }, { opacity: 0 }], { ...timing, pseudoElement: '::backdrop' }))
  } catch {
    return Promise.resolve()
  }
}

// an animation's end, whether it ran out or was stopped
const settle = (motion: Animation) =>
  motion.finished.then(
    () => undefined,
    () => undefined,
  )

// A window in the background does not run its animations, and a sheet must
// still close: the motions end, or their time runs out with some to spare.
const SPARE_MS = 100
const within = (ms: number, ...motions: Promise<void>[]) =>
  Promise.race([Promise.all(motions), new Promise<void>((done) => window.setTimeout(done, ms + SPARE_MS))])
