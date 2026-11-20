import assert from 'node:assert/strict'
import { test } from 'node:test'
import { zoomFrame, zoomKeyframes } from './zoom.ts'

const box = (x: number, y: number, width: number, height: number) => ({ x, y, width, height })

test('a wide card: the sheet shrinks to its width and is cut to its height', () => {
  // sheet 640 x 736 at (400, 100); card 320 x 160 at (80, 500)
  // what is left of the sheet, in its own pixels, is 640 x 320: the card at half size
  assert.deepEqual(zoomFrame(box(80, 500, 320, 160), box(400, 100, 640, 736)), {
    x: -320,
    y: 400,
    scale: 0.5,
    right: 0,
    bottom: 416,
  })
})

test('a tall origin: the sheet shrinks to its height and is cut to its width', () => {
  assert.deepEqual(zoomFrame(box(0, 0, 100, 368), box(0, 0, 640, 736)), {
    x: 0,
    y: 0,
    scale: 0.5,
    right: 440,
    bottom: 0,
  })
})

test('a sheet with no size does not zoom', () => {
  assert.equal(zoomFrame(box(0, 0, 100, 100), box(0, 0, 0, 0)), null)
  assert.equal(zoomFrame(box(0, 0, 0, 0), box(0, 0, 640, 736)), null)
})

test('the keyframes go from the card to the sheet, the corner kept at the card radius', () => {
  const frame = zoomFrame(box(80, 500, 320, 160), box(400, 100, 640, 736))
  assert.ok(frame)
  const [from, to] = zoomKeyframes(frame, 12, 16)
  assert.equal(from.transform, 'translate(-320px, 400px) scale(0.5)')
  // a 12px corner on screen is 24px of a sheet at half size
  assert.equal(from.clipPath, 'inset(0px 0px 416px 0px round 24px)')
  assert.equal(to.transform, 'translate(0px, 0px) scale(1)')
  assert.equal(to.clipPath, 'inset(0px 0px 0px 0px round 16px)')
})
