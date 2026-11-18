import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  hasRuntimePath,
  preferredModel,
  readStoredLayout,
  readStoredPaths,
  storeLayout,
  storePreferredModel,
  storeRuntimePaths,
} from './appStorage.ts'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'

const LAYOUT_KEY = 'recall.layout.v2'
const DEFAULT_LAYOUT = {
  sidebarWidth: 236,
  listWidth: 300,
  reportListWidth: 318,
  sidebarCollapsed: false,
  chatListWidth: 260,
  chatListClosed: false,
  citesWidth: 360,
  citesClosed: false,
}

const withLayout = (stored: unknown) => {
  pageWithStorage(fakeStorage({ [LAYOUT_KEY]: JSON.stringify(stored) }))
  return readStoredLayout()
}

test('with nothing remembered the panes take their first sizes', () => {
  pageWithStorage(fakeStorage())

  assert.deepEqual(readStoredLayout(), DEFAULT_LAYOUT)
})

test('sizes that were remembered come back', () => {
  const layout = {
    sidebarWidth: 260,
    listWidth: 340,
    reportListWidth: 400,
    sidebarCollapsed: true,
    chatListWidth: 300,
    chatListClosed: true,
    citesWidth: 420,
    citesClosed: true,
  }
  pageWithStorage(fakeStorage())

  storeLayout(layout)

  assert.deepEqual(readStoredLayout(), layout)
})

test('a size outside what a pane can be dragged to is held to the nearest end', () => {
  assert.deepEqual(withLayout({ sidebarWidth: 10, listWidth: 9999, reportListWidth: 100, sidebarCollapsed: false, chatListWidth: 50 }), {
    sidebarWidth: 220,
    listWidth: 480,
    reportListWidth: 240,
    sidebarCollapsed: false,
    chatListWidth: 200,
    chatListClosed: false,
    citesWidth: 360,
    citesClosed: false,
  })
  assert.equal(withLayout({ chatListWidth: 9999 }).chatListWidth, 420)
})

test('a size that is missing or is not a number is the first size, not the smallest', () => {
  assert.deepEqual(withLayout({ sidebarWidth: null, listWidth: '', reportListWidth: 400 }), {
    ...DEFAULT_LAYOUT,
    reportListWidth: 400,
  })
  assert.deepEqual(withLayout({ sidebarWidth: '260', listWidth: [], reportListWidth: true }), DEFAULT_LAYOUT)
  assert.deepEqual(withLayout({ sidebarWidth: Number.NaN, listWidth: Number.POSITIVE_INFINITY }), DEFAULT_LAYOUT)
  assert.deepEqual(withLayout({}), DEFAULT_LAYOUT)
})

test('the rail is folded only when that is what was remembered', () => {
  assert.equal(withLayout({ sidebarCollapsed: true }).sidebarCollapsed, true)
  assert.equal(withLayout({ sidebarCollapsed: 'false' }).sidebarCollapsed, false)
  assert.equal(withLayout({ sidebarCollapsed: 1 }).sidebarCollapsed, false)
})

test('a remembered layout that is not a layout is ignored', () => {
  ;['{{{', '7', 'null', '"wide"'].forEach((raw) => {
    pageWithStorage(fakeStorage({ [LAYOUT_KEY]: raw }))

    assert.deepEqual(readStoredLayout(), DEFAULT_LAYOUT, raw)
  })
})

test('the remembered quality is used only while the server still offers it', () => {
  pageWithStorage(fakeStorage())
  assert.equal(preferredModel(['gpt-5.4', 'gpt-5.5'], 'gpt-5.4'), 'gpt-5.4')

  storePreferredModel('gpt-5.5')

  assert.equal(preferredModel(['gpt-5.4', 'gpt-5.5'], 'gpt-5.4'), 'gpt-5.5')
  assert.equal(preferredModel(['gpt-5.4'], 'gpt-5.4'), 'gpt-5.4')
})

test('a remembered file location stands in for the one the server suggests', () => {
  const suggested = { messagesPath: '/suggested/messages.csv', outDir: '/suggested/out' }
  pageWithStorage(fakeStorage())
  assert.deepEqual(readStoredPaths(suggested), suggested)

  storeRuntimePaths({ messagesPath: '/mine/messages.csv', outDir: '' })

  assert.deepEqual(readStoredPaths(suggested), { ...suggested, messagesPath: '/mine/messages.csv' })
  assert.equal(hasRuntimePath({ messagesPath: '', outDir: '' }), false)
  assert.equal(hasRuntimePath({ messagesPath: '/mine/messages.csv', outDir: '' }), true)
})

// An update reads Messages itself, so the app keeps no path to a database.
test('a database path remembered by an earlier build is left behind', () => {
  const suggested = { messagesPath: '/suggested/messages.csv', outDir: '/suggested/out' }
  const remembered = { dbPath: '/old/chat.db', messagesPath: '', outDir: '/mine/out' }
  pageWithStorage(fakeStorage({ 'recall.runtimePaths.v1': JSON.stringify(remembered) }))

  assert.deepEqual(readStoredPaths(suggested), { messagesPath: '/suggested/messages.csv', outDir: '/mine/out' })
  // a path to a database alone is no location that was chosen
  assert.equal(hasRuntimePath({ ...remembered, outDir: '' }), false)
})

test('where nothing can be remembered the app starts from its first settings', () => {
  const suggested = { messagesPath: 'b', outDir: 'c' }
  const starts = () => {
    assert.deepEqual(readStoredLayout(), DEFAULT_LAYOUT)
    assert.equal(preferredModel(['gpt-5.4'], 'gpt-5.4'), 'gpt-5.4')
    assert.deepEqual(readStoredPaths(suggested), suggested)
    assert.doesNotThrow(() => storeLayout(DEFAULT_LAYOUT))
    assert.doesNotThrow(() => storePreferredModel('gpt-5.4'))
    assert.doesNotThrow(() => storeRuntimePaths(suggested))
  }

  pageWithNoStorage()
  starts()
  pageWithStorage(fakeStorage({}, { broken: true }))
  starts()
  noPage()
  starts()
})

test('the list of chats is closed only when that is what was remembered', () => {
  assert.equal(withLayout({ chatListClosed: true }).chatListClosed, true)
  assert.equal(withLayout({ chatListClosed: 'true' }).chatListClosed, false)
  assert.equal(withLayout({}).chatListClosed, false)
})

test('the width of the cited messages is held to what it can be dragged to, and it is put away only when remembered so', () => {
  assert.equal(withLayout({ citesWidth: 100 }).citesWidth, 280)
  assert.equal(withLayout({ citesWidth: 9999 }).citesWidth, 560)
  assert.equal(withLayout({ citesClosed: true }).citesClosed, true)
  assert.equal(withLayout({ citesClosed: 1 }).citesClosed, false)
})
