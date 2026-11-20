import assert from 'node:assert/strict'
import test from 'node:test'
import { NIGHT as SEA_NIGHT, DAY as SEA_DAY } from './backdrop/shoreline.ts'
import { DAY as SKY_DAY, NIGHT as SKY_NIGHT } from './backdrop/sky.ts'
import { GROUND, SCENES, paintScene, painter, sceneFor } from './cardScenes.ts'

const PALETTE = new Set([...SEA_DAY, ...SEA_NIGHT, ...SKY_DAY, ...SKY_NIGHT])

test('every cell of every scene is a color of the rail’s own scenes', () => {
  for (const { value } of SCENES) {
    const color = painter(value, 40, 30)
    for (let y = 0; y < 30; y += 1)
      for (let x = 0; x < 40; x += 1) assert.ok(PALETTE.has(color(x, y)), `${value} ${x},${y}`)
  }
})

test('a scene is the same picture every time it is drawn', () => {
  const once = painter('aurora', 30, 20)
  const again = painter('aurora', 30, 20)
  for (let y = 0; y < 20; y += 1) for (let x = 0; x < 30; x += 1) assert.equal(once(x, y), again(x, y))
})

test('the sunrise has its sun, the sea its moon', () => {
  const sun = painter('sunrise', 90, 60)
  assert.equal(sun(Math.round(90 * 0.3), Math.floor(60 * 0.66) - 2), SKY_DAY[6])
  const moon = painter('sea', 90, 60)
  assert.equal(moon(Math.round(90 * 0.7), Math.round(60 * 0.25)), SKY_NIGHT[6])
})

test('the water is under the horizon and the sky over it', () => {
  const night = painter('sea', 90, 60)
  assert.ok(SEA_NIGHT.includes(night(5, 59)))
  assert.ok(SKY_NIGHT.includes(night(5, 0)))
})

test('each kind of card starts on its own scene, and each scene has a night under it', () => {
  assert.equal(sceneFor('anniversary'), 'sunrise')
  assert.equal(sceneFor('onthisday'), 'sea')
  assert.equal(sceneFor('duo'), 'aurora')
  assert.deepEqual(Object.keys(GROUND).sort(), ['aurora', 'sea', 'sunrise'])
  for (const color of Object.values(GROUND)) assert.match(color, /^#[0-9a-f]{6}$/)
})

test('a scene is painted in runs that cover every cell once', () => {
  const rects: Array<[number, number, number, number]> = []
  const pen = {
    fillStyle: '' as string,
    fillRect: (x: number, y: number, w: number, h: number) => void rects.push([x, y, w, h]),
  }
  paintScene(pen, 'sunrise', 20, 10, 4)
  const area = rects.reduce((sum, [, , w, h]) => sum + w * h, 0)
  assert.equal(area, 20 * 10 * 16)
  assert.ok(rects.length < 200, 'neighbouring cells of one color are one rectangle')
})
