import assert from 'node:assert/strict'
import { test } from 'node:test'
import { rampCss, toneAt } from './tone.ts'

const RAMP = ['#000000', '#ff0000', '#ffffff']

test('the colour at each end of a ramp is its first and its last colour', () => {
  assert.equal(toneAt(RAMP, 0), 'rgb(0 0 0)')
  assert.equal(toneAt(RAMP, 1), 'rgb(255 255 255)')
})

test('between two colours of a ramp the colour is mixed by how far along it is', () => {
  assert.equal(toneAt(RAMP, 0.5), 'rgb(255 0 0)')
  assert.equal(toneAt(RAMP, 0.25), 'rgb(128 0 0)')
  assert.equal(toneAt(RAMP, 0.75), 'rgb(255 128 128)')
})

test('a place past an end is that end, and no number is the start', () => {
  assert.equal(toneAt(RAMP, -4), 'rgb(0 0 0)')
  assert.equal(toneAt(RAMP, 4), 'rgb(255 255 255)')
  assert.equal(toneAt(RAMP, Number.NaN), 'rgb(0 0 0)')
})

test('a ramp of one colour is that colour everywhere, and an empty ramp has none', () => {
  assert.equal(toneAt(['#0a84ff'], 0.7), 'rgb(10 132 255)')
  assert.equal(toneAt([], 0.7), 'transparent')
})

test('a colour that is not six hex digits is read as black, and not as a fault', () => {
  assert.equal(toneAt(['red', '#ffffff'], 0), 'rgb(0 0 0)')
})

test('a ramp as a style is its colours from left to right', () => {
  assert.equal(rampCss(RAMP), 'linear-gradient(90deg, #000000, #ff0000, #ffffff)')
  assert.equal(rampCss(['#0a84ff']), 'linear-gradient(90deg, #0a84ff, #0a84ff)')
})
