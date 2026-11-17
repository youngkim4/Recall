import assert from 'node:assert/strict'
import { test } from 'node:test'
import { scrollToShow, stepIndex, tabStopIn } from './roving.ts'

test('the arrow keys move the highlight one row and stop at the ends', () => {
  assert.equal(stepIndex('ArrowDown', 0, 3), 1)
  assert.equal(stepIndex('ArrowDown', 2, 3), 2)
  assert.equal(stepIndex('ArrowUp', 0, 3), 0)
  assert.equal(stepIndex('End', 0, 3), 2)
  assert.equal(stepIndex('Enter', 0, 3), null)
  assert.equal(stepIndex('ArrowDown', 0, 0), null)
})

// a conversation from 5 October 2023 to 20 March 2025
const FIRST = '2023-10-05'
const LAST = '2025-03-20'

test('the day Tab stops on is the first wanted day that the month holds', () => {
  assert.equal(tabStopIn('2024-06', [null, '2024-05-30', '2024-06-12', '2024-06-20'], FIRST, LAST), '2024-06-12')
})

test('a wanted day outside the conversation gives way to the nearest day inside it', () => {
  // dates kept from another conversation can end after this one does
  assert.equal(tabStopIn('2025-03', [null, null, '2025-03-31'], FIRST, LAST), '2025-03-20')
  assert.equal(tabStopIn('2023-10', ['2023-10-02'], FIRST, LAST), '2023-10-05')
})

test('a month with no wanted day stops on its first day, and the last month on the last day', () => {
  assert.equal(tabStopIn('2024-06', [null, undefined, '2024-02-01'], FIRST, LAST), '2024-06-01')
  assert.equal(tabStopIn('2023-10', [], FIRST, LAST), '2023-10-05')
  assert.equal(tabStopIn('2025-03', [], FIRST, LAST), '2025-03-20')
})

// a list 200 tall showing what lies from 300 to 500 of its rows
const VIEW = { top: 300, height: 200 }

test('a row already in view leaves the list where it is', () => {
  assert.equal(scrollToShow(VIEW, { top: 340, height: 32 }, 'nearest'), 300)
})

test('a row above or below what is in view is brought to the nearer edge', () => {
  assert.equal(scrollToShow(VIEW, { top: 280, height: 32 }, 'nearest'), 280)
  assert.equal(scrollToShow(VIEW, { top: 490, height: 32 }, 'nearest'), 322)
})

test('a row can be asked into the middle, as far as the top of the list allows', () => {
  assert.equal(scrollToShow(VIEW, { top: 1000, height: 32 }, 'center'), 916)
  assert.equal(scrollToShow(VIEW, { top: 40, height: 32 }, 'center'), 0)
})
