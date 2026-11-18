import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { paletteOptions, type Palette } from '../palette.ts'
import { contrast, over, paletteTokens, SELECTED, type Rgb } from '../paletteColors.test.ts'
import { frameOf, type Scene } from './craft.ts'
import { frameFor } from './run.ts'
import { sceneOf, scenes } from './scenes.ts'

const scenesCss = readFileSync('src/styles/scenes.css', 'utf8')

const MODES = ['light', 'dark'] as const
const each = Object.entries(scenes) as Array<[Palette, Scene]>

// a rail of 236 by 448 px in cells of 4px: its rows at the top and at the foot
const room = () => frameOf(59, 112, 59, 30, 104)
// moments spread over the slow things a scene does: a tide, a sway, a falling star
const MOMENTS = [0, 0.6, 3.3, 7.9, 18.2, 41.5]

const sameColor = (a: readonly number[], b: readonly number[]) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2]

// which of a scene's colours a cell of a painted frame has
function placeAt(frame: ReturnType<typeof room>, colors: readonly Rgb[], x: number, y: number): number {
  const at = (y * frame.cols + x) * 4
  return colors.findIndex((color) => sameColor(color, [frame.cells[at], frame.cells[at + 1], frame.cells[at + 2]]))
}

test('a palette is a scene or it is the first palette, and the style sheet lays the app over the same scenes', () => {
  const named = paletteOptions.map((option) => option.value).filter((value) => sceneOf(value))
  assert.deepEqual(named, paletteOptions.map((option) => option.value).filter((value) => value !== 'graphite'))
  assert.equal(sceneOf('graphite'), null)
  assert.equal(sceneOf('toString'), null)
  assert.equal(sceneOf(null), null)

  // every palette scenes.css names is a scene, and the rules that lay the app over a scene name them all
  const lists = [...scenesCss.matchAll(/:root:is\(([^)]+)\):not\(\[data-paper\]\)/g)].map((found) =>
    [...found[1].matchAll(/\[data-palette='([a-z]+)'\]/g)].map((name) => name[1]),
  )
  assert.ok(lists.length > 0, 'scenes.css names no scene')
  const everyName = [...scenesCss.matchAll(/\[data-palette='([a-z]+)'\]/g)].map((found) => found[1])
  for (const name of everyName) assert.ok((named as string[]).includes(name), `scenes.css names ${name}, which is no scene`)
  const whole = lists.filter((list) => list.length === named.length)
  assert.ok(whole.length >= 3, 'the rail, the page and the composer are not laid out for every scene')
  for (const list of whole) assert.deepEqual(list, named)
})

for (const [name, scene] of each) {
  for (const mode of MODES) {
    const colors = scene.colors[mode] as readonly Rgb[]

    test(`${name}, ${mode}: every cell is one of the scene's colours, and under a row of the rail it is ground`, () => {
      const frame = room()
      for (const moment of MOMENTS) {
        scene.paint(frame, moment, mode === 'dark')
        for (let y = 0; y < frame.rows; y += 1) {
          for (let x = 0; x < frame.cols; x += 1) {
            const place = placeAt(frame, colors, x, y)
            assert.notEqual(place, -1, `a colour that is not the scene's at ${x}, ${y}, time ${moment}`)
            assert.equal(frame.cells[(y * frame.cols + x) * 4 + 3], 255)
            const underRow = x < frame.railCols && (y < frame.navRows || y >= frame.footRow)
            if (underRow) assert.ok(scene.ground[mode].includes(place), `colour ${place} under a row at ${x}, ${y}, time ${moment}`)
          }
        }
      }
    })

    test(`${name}, ${mode}: a speck is one cell here and there and never an area`, () => {
      const frame = room()
      for (const moment of MOMENTS) {
        scene.paint(frame, moment, mode === 'dark')
        let specks = 0
        for (let y = 0; y < frame.rows; y += 1) {
          for (let x = 0; x < frame.cols; x += 1) {
            if (scene.specks[mode].includes(placeAt(frame, colors, x, y))) specks += 1
          }
        }
        assert.ok(specks <= frame.cols * frame.rows * 0.03, `${specks} cells of speck at time ${moment}`)
      }
    })

    test(`${name}, ${mode}: the words of the rail hold 4.5 to 1 on the scene's ground, a selected row too`, () => {
      const tokens = paletteTokens(name)
      const get = (token: string) => tokens.get(token)![mode]
      for (const place of scene.ground[mode]) {
        for (const [what, ground] of [['ground', colors[place]], ['a selected row', over(get('gray'), SELECTED[mode], colors[place] as Rgb)]] as const) {
          for (const text of ['text', 'text-2', 'text-3']) {
            const ratio = contrast(get(text), ground as Rgb)
            assert.ok(ratio >= 4.5, `--${text} on ${what} of colour ${place}: ${ratio.toFixed(2)}`)
          }
        }
      }
    })
  }
}

for (const [name, scene] of each) {
  test(`${name}: in a frame as wide as the rail, only the room between the rows holds anything but ground`, () => {
    for (const mode of MODES) {
      const frame = room()
      scene.paint(frame, 3.3, mode === 'dark')
      const colors = scene.colors[mode] as readonly Rgb[]
      let other = 0
      for (let y = 0; y < frame.rows; y += 1) {
        for (let x = 0; x < frame.cols; x += 1) {
          const place = placeAt(frame, colors, x, y)
          const underRow = y < frame.navRows || y >= frame.footRow
          if (underRow) assert.ok(scene.ground[mode].includes(place), `${mode}: colour ${place} under a row at ${x}, ${y}`)
          else if (!scene.ground[mode].includes(place)) other += 1
        }
      }
      // the room between the rows has a picture in it, and not only ground
      assert.ok(other > 0, `${mode}: nothing but ground in the rail`)
    }
  })
}

for (const [name, scene] of each) {
  test(`${name}: in the rail, every colour that fills an area shows at some moment`, () => {
    for (const mode of MODES) {
      const colors = scene.colors[mode] as readonly Rgb[]
      const seen = new Set<number>()
      for (const moment of [...MOMENTS, 60, 120, 250]) {
        const frame = scene.cell === 1 ? frameOf(236, 448, 236, 120, 416, 36) : room()
        scene.paint(frame, moment, mode === 'dark')
        for (let y = 0; y < frame.rows; y += 1) for (let x = 0; x < frame.cols; x += 1) seen.add(placeAt(frame, colors, x, y))
      }
      const missing = colors.map((_, place) => place).filter((place) => !scene.specks[mode].includes(place) && !seen.has(place))
      assert.deepEqual(missing, [], `${mode}: colours never shown in the rail`)
    }
  })
}

test('with no rail the whole picture is open; with one, its rows are closed and the room between them is open', () => {
  const bare = frameOf(40, 30, 0, 0, 30)
  assert.ok(bare.open.every((share) => share === 1))

  const frame = frameOf(120, 90, 30, 20, 80)
  const openAt = (x: number, y: number) => frame.open[y * frame.cols + x]
  assert.equal(openAt(5, 5), 0)
  assert.equal(openAt(29, 19), 0)
  assert.equal(openAt(5, 85), 0)
  assert.equal(openAt(5, 50), 1)
  assert.equal(openAt(100, 5), 1)
  // a slope between: nearer a row is never more open than further from it
  for (let y = 20; y < 50; y += 1) assert.ok(openAt(5, y) <= openAt(5, y + 1))
})

test('a frame for a room keeps a cell of ground around the rows, at the size of cell a scene asks for', () => {
  const frame = frameFor({ width: 236, height: 900, railRight: 236, navBottom: 233, footTop: 860 })
  assert.deepEqual([frame.cols, frame.rows, frame.railCols, frame.navRows, frame.footRow], [59, 225, 59, 60, 214])
  const fine = frameFor({ width: 236, height: 900, railRight: 236, navBottom: 233, footTop: 860 }, 1)
  assert.deepEqual([fine.cols, fine.rows, fine.railCols, fine.navRows, fine.footRow], [236, 900, 236, 234, 859])
  // the slope from a row to the open picture is as long in px at either size of cell
  const slope = (f: ReturnType<typeof frameFor>, cell: number) => {
    let y = f.navRows
    while (y < f.rows && f.open[y * f.cols] < 1) y += 1
    return (y - f.navRows) * cell
  }
  assert.ok(Math.abs(slope(frame, 4) - slope(fine, 1)) <= 4)

  const noRail = frameFor({ width: 800, height: 600, railRight: 0, navBottom: 0, footTop: 600 })
  assert.ok(noRail.open.every((share) => share === 1))
})
