// The colours of the app. A palette tints the grounds, the lines and the
// text, and gives the app its one hue: the colour of what you sent. Each
// palette has a light and a dark form, so it goes with any Appearance.
// 'graphite' is the app with nothing chosen. index.html puts a kept palette
// on the page before the first paint, with the same key and the same names.

export type Palette = 'graphite' | 'letter' | 'rose' | 'garden' | 'dusk' | 'ember'

const PALETTE_STORAGE_KEY = 'recall.palette'
const FIRST: Palette = 'graphite'

export const paletteOptions: Array<{ value: Palette; label: string }> = [
  { value: 'graphite', label: 'Graphite' },
  { value: 'letter', label: 'Letter' },
  { value: 'rose', label: 'Rose' },
  { value: 'garden', label: 'Garden' },
  { value: 'dusk', label: 'Dusk' },
  { value: 'ember', label: 'Ember' },
]

function isPalette(value: unknown): value is Palette {
  return paletteOptions.some((option) => option.value === value)
}

export function readPalette(): Palette {
  try {
    const stored = window.localStorage.getItem(PALETTE_STORAGE_KEY)
    return isPalette(stored) ? stored : FIRST
  } catch {
    return FIRST
  }
}

// what a palette is put on: the page, or what stands in for it in a test
type Root = { dataset: Record<string, string | undefined> }

export function applyPalette(palette: Palette, root: Root = document.documentElement) {
  if (palette === FIRST) {
    delete root.dataset.palette
  } else {
    root.dataset.palette = palette
  }
}

export function storePalette(palette: Palette) {
  try {
    if (palette === FIRST) {
      window.localStorage.removeItem(PALETTE_STORAGE_KEY)
    } else {
      window.localStorage.setItem(PALETTE_STORAGE_KEY, palette)
    }
  } catch {
    // the palette still applies for this session
  }
}
