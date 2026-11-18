import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fakeStorage, noPage, pageWithNoStorage, pageWithStorage } from './fakeStorage.test.ts'
import { applyPalette, paletteOptions, readPalette, storePalette, type Palette } from './palette.ts'

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

  storePalette('rose')
  assert.equal(storage.getItem(KEY), 'rose')
  assert.equal(readPalette(), 'rose')

  storePalette('graphite')
  assert.equal(storage.getItem(KEY), null)
  assert.equal(readPalette(), 'graphite')
})

test('a kept word that is no palette is read as the first palette', () => {
  for (const kept of ['', 'Rose', 'neon', '{"a":1}']) {
    pageWithStorage(fakeStorage({ [KEY]: kept }))
    assert.equal(readPalette(), 'graphite', kept)
  }
})

test('with no page or no storage the palette is the first one, and keeping one does not throw', () => {
  noPage()
  assert.equal(readPalette(), 'graphite')
  storePalette('dusk')
  pageWithNoStorage()
  assert.equal(readPalette(), 'graphite')
  storePalette('dusk')
})

test('a palette is put on the page as data-palette; the first one takes it off', () => {
  const root = { dataset: {} as Record<string, string | undefined> }

  applyPalette('garden', root)
  assert.equal(root.dataset.palette, 'garden')
  applyPalette('graphite', root)
  assert.equal('palette' in root.dataset, false)
})

// ---- the colours: each text step is read on each ground it can stand on ----

type Rgb = [number, number, number]
type Pair = { light: Rgb; dark: Rgb }

const hex = (value: string): Rgb => [1, 3, 5].map((at) => parseInt(value.slice(at, at + 2), 16)) as Rgb

function colorOf(value: string): Rgb {
  const spaced = /^rgb\((\d+) (\d+) (\d+)\)$/.exec(value)
  if (spaced) return [Number(spaced[1]), Number(spaced[2]), Number(spaced[3])]
  assert.match(value, /^#[0-9a-f]{6}$/, `a colour of a palette is six hex digits or rgb(r g b): ${value}`)
  return hex(value)
}

// the colour tokens of one block: `--name: light-dark(a, b);` or `--name: a;`
function tokensOf(block: string): Map<string, Pair> {
  const found = new Map<string, Pair>()
  for (const [, name, value] of block.matchAll(/--([a-z0-9-]+):\s*([^;]+);/g)) {
    const both = /^light-dark\((#[0-9a-f]{6}|rgb\([\d ]+\)),\s*(#[0-9a-f]{6}|rgb\([\d ]+\))\)$/.exec(value.trim())
    if (both) found.set(name, { light: colorOf(both[1]), dark: colorOf(both[2]) })
    else if (/^(#[0-9a-f]{6}|rgb\([\d ]+\))$/.test(value.trim())) {
      found.set(name, { light: colorOf(value.trim()), dark: colorOf(value.trim()) })
    }
  }
  return found
}

function blockAfter(css: string, opening: string): string {
  const from = css.indexOf(opening)
  assert.notEqual(from, -1, `no block opens with ${opening}`)
  return css.slice(from, css.indexOf('}', from))
}

const tokensCss = readFileSync('src/styles/tokens.css', 'utf8')
const palettesCss = readFileSync('src/styles/palettes.css', 'utf8')
const graphite = tokensOf(blockAfter(tokensCss, ':root,'))

function paletteTokens(palette: Palette): Map<string, Pair> {
  if (palette === 'graphite') return graphite
  // a palette changes some tokens of the first palette and keeps the rest
  return new Map([...graphite, ...tokensOf(blockAfter(palettesCss, `[data-palette='${palette}']`))])
}

function luminance([r, g, b]: Rgb) {
  const part = (value: number) => {
    const share = value / 255
    return share <= 0.04045 ? share / 12.92 : ((share + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * part(r) + 0.7152 * part(g) + 0.0722 * part(b)
}

function contrast(a: Rgb, b: Rgb) {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (high + 0.05) / (low + 0.05)
}

const over = (top: Rgb, share: number, under: Rgb): Rgb =>
  top.map((part, index) => part * share + under[index] * (1 - share)) as Rgb

// how much of --gray a selected row lays over its ground (--fill-active in tokens.css)
const SELECTED = { light: 0.16, dark: 0.28 } as const

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
      assert.ok(contrast([255, 255, 255], get('you')) >= 4.5, 'white on --you')
      assert.ok(contrast(get('on-ink'), get('ink')) >= 4.5, '--on-ink on --ink')
      assert.ok(contrast(get('text'), get('them')) >= 4.5, '--text on --them')
    })
  }
}

test('the style sheet has one block for each palette after the first, and no other', () => {
  const written = [...palettesCss.matchAll(/\[data-palette='([a-z]+)'\]/g)].map((found) => found[1])
  const wanted = paletteOptions.map((option) => option.value).filter((value) => value !== 'graphite')

  assert.deepEqual([...new Set(written)].sort(), [...wanted].sort())
})

test('the page takes a kept palette before it is drawn, with the same key and the same names', () => {
  const page = readFileSync('index.html', 'utf8')

  assert.ok(page.includes(`localStorage.getItem('${KEY}')`))
  for (const { value } of paletteOptions) {
    if (value !== 'graphite') assert.ok(page.includes(`'${value}'`), value)
  }
})
