import assert from 'node:assert/strict'
import { test } from 'node:test'
import { DRAG_ZONE, pressOf } from './windowDrag.ts'

// what stands in for the thing pressed: it is, or is inside, something that takes a press, or not
const bare = { closest: () => null }
const control = { closest: () => ({}) }

test('a press on the bare top of the window moves it; a double press zooms it', () => {
  assert.equal(pressOf({ button: 0, clientY: 10, detail: 1, target: bare }), 'drag')
  assert.equal(pressOf({ button: 0, clientY: 10, detail: 2, target: bare }), 'zoom')
  assert.equal(pressOf({ button: 0, clientY: DRAG_ZONE - 1, detail: 1, target: bare }), 'drag')
})

test('a press under the top of the window does nothing to the window', () => {
  assert.equal(pressOf({ button: 0, clientY: DRAG_ZONE, detail: 1, target: bare }), 'none')
  assert.equal(pressOf({ button: 0, clientY: 400, detail: 2, target: bare }), 'none')
})

test('the parts that take a press are named: controls, words, and anything marked', () => {
  const asked: string[] = []
  const recorder = { closest: (selector: string) => (asked.push(selector), null) }
  pressOf({ button: 0, clientY: 10, detail: 1, target: recorder })
  const named = asked.join(', ')
  for (const part of ['button', 'input', '[role]', 'p', 'h2', 'dt', 'span', '.msg', '[data-no-drag]']) assert.ok(named.includes(part), part)
})

test('a press on a control at the top is the control\'s, and so is a press of another button', () => {
  assert.equal(pressOf({ button: 0, clientY: 10, detail: 1, target: control }), 'none')
  assert.equal(pressOf({ button: 2, clientY: 10, detail: 1, target: bare }), 'none')
  assert.equal(pressOf({ button: 0, clientY: 10, detail: 1, target: null }), 'none')
})
