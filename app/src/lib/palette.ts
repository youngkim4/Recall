// The looks of the app. 'graphite' is the app with nothing chosen: plain
// grounds and one blue. Each of the others is a scene: a picture that moves
// in the rail (lib/backdrop/), with the colours, the face and the shape of
// the page beside it (palettes.css). Each has a day and a night, so it goes with
// any Appearance. index.html puts a kept palette on the page before the
// first paint, with the same key and the same names.

export type Palette = 'graphite' | 'broadsheet' | 'shoreline' | 'canopy' | 'sky'

const PALETTE_STORAGE_KEY = 'recall.palette'
const FIRST: Palette = 'graphite'

export const paletteOptions: Array<{ value: Palette; label: string }> = [
  { value: 'graphite', label: 'Graphite' },
  { value: 'broadsheet', label: 'Broadsheet' },
  { value: 'shoreline', label: 'Shoreline' },
  { value: 'canopy', label: 'Canopy' },
  { value: 'sky', label: 'Sky' },
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
