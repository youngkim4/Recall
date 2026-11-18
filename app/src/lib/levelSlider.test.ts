import assert from 'node:assert/strict'
import { test } from 'node:test'
import { alongAt, heldStep, stepAt, stepByKey } from './levelSlider.ts'

test('a point on the track is the nearest step; the steps stand an inset in from each end', () => {
  // a track of 220px from x=100, the first step 10px in and the last 10px from the end: 5 steps, 50px apart
  const at = (x: number) => stepAt(x, { left: 100, width: 220 }, 10, 5)

  assert.equal(at(110), 0)
  assert.equal(at(134), 0)
  assert.equal(at(136), 1)
  assert.equal(at(210), 2)
  assert.equal(at(310), 4)
})

test('a point past an end is the step at that end', () => {
  const at = (x: number) => stepAt(x, { left: 100, width: 220 }, 10, 5)

  assert.equal(at(-500), 0)
  assert.equal(at(100), 0)
  assert.equal(at(5000), 4)
})

test('a track with one step, or with no width, has only its first step', () => {
  assert.equal(stepAt(150, { left: 100, width: 220 }, 10, 1), 0)
  assert.equal(stepAt(150, { left: 100, width: 0 }, 10, 5), 0)
  assert.equal(stepAt(150, { left: 100, width: 20 }, 10, 5), 0)
  assert.equal(stepAt(150, { left: 100, width: 220 }, 10, 0), 0)
})

test('the arrow keys move one step and the ends hold', () => {
  assert.equal(stepByKey('ArrowRight', 1, 5), 2)
  assert.equal(stepByKey('ArrowUp', 1, 5), 2)
  assert.equal(stepByKey('ArrowLeft', 1, 5), 0)
  assert.equal(stepByKey('ArrowDown', 1, 5), 0)
  assert.equal(stepByKey('ArrowLeft', 0, 5), 0)
  assert.equal(stepByKey('ArrowRight', 4, 5), 4)
})

test('Home and End go to the ends', () => {
  assert.equal(stepByKey('Home', 3, 5), 0)
  assert.equal(stepByKey('End', 1, 5), 4)
})

test('another key moves nothing, and so does a track with no steps', () => {
  assert.equal(stepByKey('Enter', 1, 5), null)
  assert.equal(stepByKey('a', 1, 5), null)
  assert.equal(stepByKey('ArrowRight', 0, 0), null)
})

test('a point on the track is a share of the way from the first step to the last', () => {
  // 220px from x=100, the steps 10px in from each end: 200px of reach
  const at = (x: number) => alongAt(x, { left: 100, width: 220 }, 10)

  assert.equal(at(110), 0)
  assert.equal(at(160), 0.25)
  assert.equal(at(210), 0.5)
  assert.equal(at(310), 1)
})

test('a point past an end is that end, and a track with no reach is its start', () => {
  assert.equal(alongAt(-500, { left: 100, width: 220 }, 10), 0)
  assert.equal(alongAt(5000, { left: 100, width: 220 }, 10), 1)
  assert.equal(alongAt(150, { left: 100, width: 20 }, 10), 0)
  assert.equal(alongAt(150, { left: 100, width: 0 }, 10), 0)
})

test('a held knob keeps its step until the pointer is well past half way to the next', () => {
  // 5 steps: 0, 0.25, 0.5, 0.75, 1. From step 1 (0.25), half way to step 2 is 0.375.
  assert.equal(heldStep(0.3, 5, 1), 1)
  assert.equal(heldStep(0.375, 5, 1), 1)
  // just past half way it still holds: a hand that rests there does not flip between two steps
  assert.equal(heldStep(0.39, 5, 1), 1)
  assert.equal(heldStep(0.42, 5, 1), 2)
})

test('a step that was just taken is not given back by a small move the other way', () => {
  // on step 2 (0.5) after coming up from step 1: back at 0.39 it stays on 2
  assert.equal(heldStep(0.39, 5, 2), 2)
  assert.equal(heldStep(0.33, 5, 2), 1)
})

test('a fast move goes straight to the step under the pointer', () => {
  assert.equal(heldStep(1, 5, 0), 4)
  assert.equal(heldStep(0.76, 5, 0), 3)
  assert.equal(heldStep(-2, 5, 3), 0)
  assert.equal(heldStep(9, 5, 1), 4)
})

test('with one step or none the held step is the first', () => {
  assert.equal(heldStep(0.7, 1, 0), 0)
  assert.equal(heldStep(0.7, 0, 0), 0)
})
