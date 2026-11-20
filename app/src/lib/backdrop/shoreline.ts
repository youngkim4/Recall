// A shore seen from above. Sand lies under the rail and the sea under the
// rest. The water's edge comes in and goes out; in the empty middle of the
// rail it turns in to make a cove. Lines of swell cross the water, an older
// line of foam breaks up behind the new one, and the sun or the moon glints.
import { chance, dithered, put, type Frame, type Rgb, type Scene } from './craft.ts'

// deep, open water, shallows, foam, wet sand, sand, dry sand
export const DAY: Rgb[] = [[13, 92, 128], [22, 134, 162], [72, 186, 194], [246, 251, 246], [205, 182, 138], [236, 217, 174], [245, 232, 198]]
export const NIGHT: Rgb[] = [[5, 16, 30], [9, 34, 56], [18, 66, 92], [190, 222, 234], [30, 34, 46], [42, 44, 58], [52, 52, 66]]
const SAND = 5

function paint(frame: Frame, time: number, dark: boolean) {
  const colors = dark ? NIGHT : DAY
  const { cols, rows, railCols, navRows, footRow, open } = frame
  const sea = cols - railCols
  // how far out the water is deep: the sea beside the rail, or most of the cove inside it
  const reach = Math.max(6, sea > 0 ? sea : railCols * 0.55)
  // the tide: the whole edge comes in and goes out, slowly
  const tide = 2.6 * Math.sin(time * 0.38)
  const coveAt = (navRows + footRow) / 2
  const coveTall = Math.max(1, (footRow - navRows) / 2)
  for (let y = 0; y < rows; y += 1) {
    const down = y / rows
    // the cove: the edge turns in to the middle of the rail and out again
    const inCove = Math.max(0, 1 - ((y - coveAt) / coveTall) ** 2)
    const cove = railCols * 0.78 * inCove * inCove
    const still = railCols + sea * (0.11 + 0.05 * Math.sin(down * 5.2 + 0.6)) - cove
    const edge = still + tide + 1.6 * Math.sin(y * 0.11 + time * 0.5) + 0.9 * Math.sin(y * 0.29 - time * 0.8)
    for (let x = 0; x < cols; x += 1) {
      // cells out to sea from the edge of the water
      const out = x - edge
      const clear = open[y * cols + x]
      if (out > 0 && clear > 0) {
        const far = out / reach
        const swell = 0.03 * Math.sin(out * 0.42 - time * 1.1 + Math.sin(y * 0.07) * 1.6)
        // the new line of foam, the broken older one behind it, and glints
        const foam = out < 1.6 || (Math.abs(out - 7.5 - 1.5 * Math.sin(y * 0.2 + time * 0.3)) < 1.1 && chance(x, y, Math.floor(time * 0.6)) > 0.42)
        const glint = far > 0.16 && chance(x, y, Math.floor(time * 1.4)) > 0.9945
        put(frame, x, y, foam || glint ? colors[3] : colors[2 - dithered((far + swell) * 4.4 + 0.15, x, y, 2)])
      } else {
        // wet where the water has just been, then sand, then dry sand in long ripples
        const up = -out
        const ripple = 0.5 + 0.5 * Math.sin((x * 0.5 + y) * 0.21 + Math.sin(x * 0.05) * 2)
        const shell = clear === 1 && up > 6 && chance(x, y, 7) > 0.9972
        const ground = up < 5 && clear > 0 ? 4 + dithered(up / 5, x, y, 1) : SAND + dithered(ripple * 1.5 - 0.5, x, y, 1)
        put(frame, x, y, shell ? colors[3] : colors[ground])
      }
    }
  }
}

export const shoreline: Scene = { colors: { light: DAY, dark: NIGHT }, ground: { light: [5, 6], dark: [5, 6] }, specks: { light: [], dark: [] }, pace: 10, paint }
