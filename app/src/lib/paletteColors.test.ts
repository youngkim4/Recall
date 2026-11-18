// What the tests of the palettes and of the scenes share: the colours a
// style sheet gives a palette, and how well one colour reads on another.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import type { Palette } from './palette.ts'

export type Rgb = [number, number, number]
export type Pair = { light: Rgb; dark: Rgb }

const hex = (value: string): Rgb => [1, 3, 5].map((at) => parseInt(value.slice(at, at + 2), 16)) as Rgb

function colorOf(value: string): Rgb {
  const spaced = /^rgb\((\d+) (\d+) (\d+)\)$/.exec(value)
  if (spaced) return [Number(spaced[1]), Number(spaced[2]), Number(spaced[3])]
  assert.match(value, /^#[0-9a-f]{6}$/, `a colour of a palette is six hex digits or rgb(r g b): ${value}`)
  return hex(value)
}

// the colour tokens of one block: `--name: light-dark(a, b);` or `--name: a;`
export function tokensOf(block: string): Map<string, Pair> {
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

export function blockAfter(css: string, opening: string): string {
  const from = css.indexOf(opening)
  assert.notEqual(from, -1, `no block opens with ${opening}`)
  return css.slice(from, css.indexOf('}', from))
}

export const tokensCss = readFileSync('src/styles/tokens.css', 'utf8')
export const palettesCss = readFileSync('src/styles/palettes.css', 'utf8')
const graphite = tokensOf(blockAfter(tokensCss, ':root,'))

export function paletteTokens(palette: Palette): Map<string, Pair> {
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

export function contrast(a: Rgb, b: Rgb) {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (high + 0.05) / (low + 0.05)
}

export const over = (top: Rgb, share: number, under: Rgb): Rgb =>
  top.map((part, index) => part * share + under[index] * (1 - share)) as Rgb

// how much of --gray a selected row lays over its ground (--fill-active in tokens.css)
export const SELECTED = { light: 0.16, dark: 0.28 } as const
