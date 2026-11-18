import { useEffect, useRef } from 'react'
import { LAMP, lampFrame, paintLamp, type LampColors, type Rgb } from '../../lib/lamp'
import { usePageAttribute, usePageIsDark } from '../../lib/usePage'

type LampProps = {
  // 0 on the first step, 1 on the last
  level: number
  // the last step: the lamp burns full and sparks rise from it
  max: boolean
  // false on a step that asks for no thinking at all: the candle is out
  lit: boolean
  // the size is the style sheet's: "lamp" beside the slider, a smaller one beside a step of work
  className?: string
}

// the lamp is drawn this many times a second; a flame needs no more
const PACE = 15
// how long the flame flares when the step changes, in ms
const FLARE_MS = 450

// a colour as the page has it now, read back from a CSS colour the element is given
function colorOf(element: HTMLElement, value: string): Rgb {
  element.style.color = value
  const read = getComputedStyle(element).color
  element.style.removeProperty('color')
  const parts = read.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0]
  // color(srgb r g b) gives shares of 1; rgb(r, g, b) gives 0 to 255
  const scale = read.startsWith('color(') ? 255 : 1
  return [parts[0] * scale, parts[1] * scale, parts[2] * scale].map(Math.round) as unknown as Rgb
}

function colorsOf(element: HTMLElement): LampColors {
  const middle = colorOf(element, 'var(--you-far)')
  return {
    wax: colorOf(element, 'var(--them)'),
    edge: colorOf(element, 'var(--text-3)'),
    outer: colorOf(element, 'var(--you)'),
    middle,
    heart: middle.map((part) => Math.round(part + (255 - part) * 0.55)) as unknown as Rgb,
  }
}

// The lamp beside the reasoning control: a candle drawn in cells whose
// flame is taller and brighter the more the model is asked to think, and
// burns full with sparks on the last step (lib/lamp.ts). On a step that asks
// for no thinking the candle is out and smokes. It flares for a moment when
// the step changes. With motion reduced it is one still frame.
export function Lamp({ level, max, lit, className = 'lamp' }: LampProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const flareFrom = useRef(0)
  const palette = usePageAttribute('data-palette')
  const dark = usePageIsDark()

  // a new step flares; the first drawing does not
  const first = useRef(true)
  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    flareFrom.current = performance.now()
  }, [level])

  useEffect(() => {
    const element = canvas.current
    const context = element?.getContext('2d')
    if (!element || !context) return
    const pen = context
    const colors = colorsOf(element)
    const frame = lampFrame()
    const image = new ImageData(frame.cells, frame.cols, frame.rows)
    const still = window.matchMedia('(prefers-reduced-motion: reduce)')

    function draw() {
      const now = performance.now()
      const flare = still.matches ? 0 : Math.max(0, 1 - (now - flareFrom.current) / FLARE_MS)
      paintLamp(frame, level, max, still.matches ? 1 : now / 1000, flare, colors, lit)
      pen.putImageData(image, 0, 0)
    }

    draw()
    const timer = window.setInterval(() => {
      if (!document.hidden && !still.matches) draw()
    }, 1000 / PACE)
    still.addEventListener('change', draw)
    return () => {
      window.clearInterval(timer)
      still.removeEventListener('change', draw)
    }
  }, [level, max, lit, palette, dark])

  return <canvas ref={canvas} className={className} width={LAMP.cols} height={LAMP.rows} aria-hidden="true" />
}
