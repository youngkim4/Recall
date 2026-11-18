import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'
import { applyPalette, paletteOptions, readPalette, storePalette } from './palette.ts'
import { blockAfter, contrast, over, palettesCss, paletteTokens, SELECTED, tokensCss, tokensOf } from './paletteColors.test.ts'

const KEY = 'recall.palette'

// ---- the preference ----

test('the first palette is the one a page has with nothing chosen', () => {
  pageWithStorage(fakeStorage())

  assert.equal(paletteOptions[0].value, 'graphite')
  assert.equal(readPalette(), 'graphite')
})

test('a chosen palette is kept and read back; the first one is kept as nothing', () => {
  const storage = fakeStorage()
  pageWithStorage(storage)

  storePalette('canopy')
  assert.equal(storage.getItem(KEY), 'canopy')
  assert.equal(readPalette(), 'canopy')

  storePalette('graphite')
  assert.equal(storage.getItem(KEY), null)
  assert.equal(readPalette(), 'graphite')
})

test('a kept word that is no palette is read as the first palette', () => {
  for (const kept of ['', 'Canopy', 'nord', '{"a":1}']) {
    pageWithStorage(fakeStorage({ [KEY]: kept }))
    assert.equal(readPalette(), 'graphite', kept)
  }
})

test('with no page or no storage the palette is the first one, and keeping one does not throw', () => {
  noPage()
  assert.equal(readPalette(), 'graphite')
  storePalette('sky')
  pageWithNoStorage()
  assert.equal(readPalette(), 'graphite')
  storePalette('sky')
})

test('a palette is put on the page as data-palette; the first one takes it off', () => {
  const root = { dataset: {} as Record<string, string | undefined> }

  applyPalette('shoreline', root)
  assert.equal(root.dataset.palette, 'shoreline')
  applyPalette('graphite', root)
  assert.equal('palette' in root.dataset, false)
})

// ---- the colours: each text step is read on each ground it can stand on ----

for (const { value: palette } of paletteOptions) {
  for (const mode of ['light', 'dark'] as const) {
    test(`${palette}, ${mode}: every text step holds 4.5 to 1 on every ground, a selected row too`, () => {
      const tokens = paletteTokens(palette)
      const get = (name: string) => {
        const pair = tokens.get(name)
        assert.ok(pair, `${palette} has no --${name}`)
        return pair[mode]
      }
      const grounds = ['bg', 'bg-rail', 'bg-raised'].flatMap((name) => [
        [name, get(name)] as const,
        [`${name}, selected`, over(get('gray'), SELECTED[mode], get(name))] as const,
      ])
      for (const text of ['text', 'text-2', 'text-3']) {
        for (const [name, ground] of grounds) {
          const ratio = contrast(get(text), ground)
          assert.ok(ratio >= 4.5, `--${text} on ${name}: ${ratio.toFixed(2)}`)
        }
      }
      for (const name of ['bg', 'bg-rail', 'bg-raised']) {
        const ratio = contrast(get('you-ink'), get(name))
        assert.ok(ratio >= 4.5, `--you-ink on ${name}: ${ratio.toFixed(2)}`)
      }
      assert.ok(contrast(get('on-you'), get('you')) >= 4.5, '--on-you on --you')
      // a bar of the hue is seen on every ground; in the scenes it is a deep ink, not a bright fill
      for (const name of ['bg', 'bg-rail', 'bg-raised']) {
        const ratio = contrast(get('you'), get(name))
        assert.ok(ratio >= 2.6, `--you on ${name}: ${ratio.toFixed(2)}`)
      }
      // a hover is a moment, and large or bold words: 3 to 1 is the rule for those
      assert.ok(contrast(get('on-you'), get('you-hover')) >= 3, '--on-you on --you-hover')
      assert.ok(contrast(get('on-ink'), get('ink')) >= 4.5, '--on-ink on --ink')
      assert.ok(contrast(get('on-ink'), get('ink-hover')) >= 4.5, '--on-ink on --ink-hover')
      assert.ok(contrast(get('text'), get('them')) >= 4.5, '--text on --them')
    })
  }
}

// what a palette must say itself; a token it left out would be Graphite's, in the wrong tint
const OWN = [
  'bg', 'bg-rail', 'bg-raised', 'gray', 'edge', 'text', 'text-2', 'text-3', 'text-off',
  'ink', 'ink-hover', 'on-ink', 'you', 'you-hover', 'on-you', 'you-ink', 'them', 'them-mark',
]

test('each palette after the first sets every colour that is a palette\'s own', () => {
  for (const { value } of paletteOptions) {
    if (value === 'graphite') continue
    const own = tokensOf(blockAfter(palettesCss, `[data-palette='${value}']`))
    for (const name of OWN) assert.ok(own.has(name), `${value} does not set --${name}`)
  }
})

test('the share of gray that a selected row lays down is the one this file measures with', () => {
  const written = /--fill-active:\s*light-dark\(color-mix\(in srgb, var\(--gray\) (\d+)%, transparent\), color-mix\(in srgb, var\(--gray\) (\d+)%, transparent\)\)/.exec(tokensCss)

  assert.ok(written, '--fill-active is not written as two shares of --gray')
  assert.deepEqual([Number(written[1]) / 100, Number(written[2]) / 100], [SELECTED.light, SELECTED.dark])
})

test('the style sheet has one block for each palette after the first, and no other', () => {
  const written = [...palettesCss.matchAll(/\[data-palette='([a-z]+)'\]/g)].map((found) => found[1])
  const wanted = paletteOptions.map((option) => option.value).filter((value) => value !== 'graphite')

  assert.deepEqual([...new Set(written)].sort(), [...wanted].sort())
})

test('the page takes a kept palette before it is drawn, with the same key and the same names', () => {
  const page = readFileSync('index.html', 'utf8')

  assert.ok(page.includes(`localStorage.getItem('${KEY}')`))
  const listed = /\[([^\]]+)\]\.indexOf\(palette\)/.exec(page)
  assert.ok(listed, 'the page has no list of palettes')
  const names = [...listed[1].matchAll(/'([a-z]+)'/g)].map((found) => found[1])
  assert.deepEqual(names, paletteOptions.map((option) => option.value).filter((value) => value !== 'graphite'))
})
