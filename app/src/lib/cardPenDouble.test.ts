// A stand-in for a canvas, for the tests of the share cards. It is not a
// test; the suffix keeps it out of the app's build.

export type Text = { text: string; left: number; right: number; base: number; size: number; color: string }
export type Box = { left: number; right: number; top: number; bottom: number; color: string }
type Point = [x: number, y: number]

const sizeOf = (font: string) => Number(/([\d.]+)px/.exec(font)?.[1] ?? 0)
const boxAround = (points: Point[], color: string): Box => ({
  left: Math.min(...points.map(([x]) => x)),
  right: Math.max(...points.map(([x]) => x)),
  top: Math.min(...points.map(([, y]) => y)),
  bottom: Math.max(...points.map(([, y]) => y)),
  color,
})

// A pen that draws nothing and remembers everything. Each letter is half as
// wide as its type is tall, so a width is a count of letters.
export function fakePen(hasRoundRect = true) {
  const texts: Text[] = []
  const boxes: Box[] = []
  const state = { path: [] as Point[] }
  const extend = (...points: Point[]) => {
    state.path = [...state.path, ...points]
  }
  const pen = {
    font: '',
    fillStyle: '',
    textAlign: 'start',
    textBaseline: 'alphabetic',
    measureText: (text: string) => ({ width: [...text].length * sizeOf(pen.font) * 0.5 }),
    fillText(text: string, x: number, y: number) {
      const width = pen.measureText(text).width
      const left = pen.textAlign === 'center' ? x - width / 2 : pen.textAlign === 'right' ? x - width : x
      texts.push({ text, left, right: left + width, base: y, size: sizeOf(pen.font), color: pen.fillStyle })
    },
    fillRect: (x: number, y: number, w: number, h: number) => boxes.push(boxAround([[x, y], [x + w, y + h]], pen.fillStyle)),
    beginPath() {
      state.path = []
    },
    moveTo: (x: number, y: number) => extend([x, y]),
    lineTo: (x: number, y: number) => extend([x, y]),
    arcTo: (x1: number, y1: number, x2: number, y2: number) => extend([x1, y1], [x2, y2]),
    closePath() {},
    fill() {
      if (state.path.length) boxes.push(boxAround(state.path, pen.fillStyle))
    },
    ...(hasRoundRect ? { roundRect: (x: number, y: number, w: number, h: number) => extend([x, y], [x + w, y + h]) } : {}),
  }
  return { pen, texts, boxes }
}

export type FakePen = ReturnType<typeof fakePen>
// the pen as the drawing code takes it
export const asPen = (double: FakePen) => double.pen as unknown as CanvasRenderingContext2D
