// The sky. By night: stars that come and go, a slow curtain of northern
// light, and now and then a star that falls. By day: the first light, a low
// sun, and banks of cloud that drift across.
import { chance, dithered, drift, put, type Frame, type Rgb, type Scene } from './craft.ts'

// night: zenith, mid sky, low sky, green of the curtain, violet of the curtain, faint star, star
export const NIGHT: Rgb[] = [[6, 9, 22], [11, 17, 40], [20, 28, 62], [30, 132, 118], [104, 86, 196], [140, 150, 204], [244, 246, 255]]
// day: high blue, pale blue, rose, peach by the horizon, shade of a cloud, cloud, sun
export const DAY: Rgb[] = [[158, 192, 238], [198, 216, 244], [240, 208, 220], [250, 224, 200], [226, 222, 240], [255, 255, 255], [255, 206, 122]]

function paintNight(frame: Frame, time: number) {
  const { cols, rows, open } = frame
  // a star falls every 18 seconds, for a little over one of them
  const turn = Math.floor(time / 18)
  const fall = (time % 18) / 1.3
  const [fromX, fromY] = [(chance(turn, 1, 9) * 0.5 + 0.35) * cols, (chance(turn, 2, 9) * 0.25 + 0.06) * rows]
  const [headX, headY] = [fromX + fall * 0.2 * cols, fromY + fall * 0.15 * rows]
  for (let x = 0; x < cols; x += 1) {
    const across = x / cols
    // The curtain hangs from a line that wanders. It is made of rays: each
    // column has its own strength and its own length, and both change slowly.
    const line = 0.3 + 0.08 * Math.sin(across * 5 + time * 0.14) + 0.035 * Math.sin(across * 12 - time * 0.2)
    const ray = drift(x * 0.3 + time * 0.12, time * 0.05, 1)
    const reach = 3.2 + 9 * drift(x * 0.11 - time * 0.07, time * 0.04, 3)
    const fold = 0.5 + 0.5 * Math.sin(across * 9 + time * 0.3 + Math.sin(across * 4 - time * 0.13) * 2)
    for (let y = 0; y < rows; y += 1) {
      const down = y / rows
      const clear = open[y * cols + x]
      let color = NIGHT[dithered(down * 2.3 - 0.15, x, y, 2)]
      const under = down - line
      if (under > -0.05 && under < 0.5 && clear > 0) {
        // soft above the line, bright just under it, fading down the length of the ray
        const height = under < 0 ? 1 + under / 0.05 : Math.exp(-under * reach)
        const strength = (0.15 + 0.85 * ray * ray) * (0.45 + 0.55 * fold) * height * clear
        const lit = dithered(strength * 2.6 - 0.12, x, y, 2)
        if (lit) color = NIGHT[lit === 2 && under < 0.08 ? 4 : 3]
      }
      const star = chance(x, y, 5)
      if (star > 0.9885 && clear === 1) {
        const shine = Math.sin(time * (0.35 + star * 1.8) + star * 700)
        if (shine > -0.35) color = NIGHT[shine > 0.6 ? 6 : 5]
      }
      if (fall < 1 && clear === 1) {
        // the falling star and the few cells of its tail
        const back = (headX - x) * 0.8 + (headY - y) * 0.6
        const off = Math.abs((headX - x) * 0.6 - (headY - y) * 0.8)
        if (back >= 0 && back < 9 && off < 0.7) color = NIGHT[back < 2 ? 6 : 5]
      }
      put(frame, x, y, color)
    }
  }
}

// how much cloud is over a cell: more than 0 is cloud
function cloud(x: number, y: number, rows: number, time: number): number {
  const bank = drift(x * 0.018 + time * 0.03, y * 0.07, 2) * 0.62 + drift(x * 0.05 + time * 0.05, y * 0.16, 4) * 0.38
  return bank - 0.53 - Math.abs(y / rows - 0.42) * 0.3
}

function paintDay(frame: Frame, time: number) {
  const { cols, rows, open } = frame
  const [sunX, sunY, sunWide] = [cols * 0.8, rows * 0.72, Math.min(cols, rows) * 0.085]
  for (let y = 0; y < rows; y += 1) {
    const down = y / rows
    for (let x = 0; x < cols; x += 1) {
      const clear = open[y * cols + x]
      const fromSun = Math.hypot(x - sunX, y - sunY) / sunWide
      let color = DAY[dithered(down * 3.5 - 0.25, x, y, 3)]
      // the sun stands clear: cloud thins out toward it, and toward a row of the rail
      const thin = (1 - clear) * 0.5 + Math.max(0, 1 - fromSun / 2.6) * 0.5
      // cloud drifts across in long flat banks, each with a shaded underside
      if (cloud(x, y, rows, time) - thin > 0) color = DAY[cloud(x, y - 3, rows, time) - thin > 0.012 ? 5 : 4]
      if (clear === 1 && fromSun < 1) color = DAY[6]
      // a ring of warm light around the sun, woven into the sky
      else if (clear === 1 && fromSun < 1.5 && dithered((1.5 - fromSun) * 2, x, y, 1)) color = DAY[3]
      put(frame, x, y, color)
    }
  }
}

function paint(frame: Frame, time: number, dark: boolean) {
  if (dark) paintNight(frame, time)
  else paintDay(frame, time)
}

export const sky: Scene = { colors: { light: DAY, dark: NIGHT }, ground: { light: [0, 1, 2, 3], dark: [0, 1, 2] }, specks: { light: [], dark: [5, 6] }, pace: 10, paint }
