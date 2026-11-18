// Which palettes have a picture behind them, and which picture.
import type { Palette } from '../palette.ts'
import { canopy } from './canopy.ts'
import type { Scene } from './craft.ts'
import { halftone } from './halftone.ts'
import { shoreline } from './shoreline.ts'
import { sky } from './sky.ts'

export const scenes: Partial<Record<Palette, Scene>> = { broadsheet: halftone, shoreline, canopy, sky }

export function sceneOf(palette: string | null | undefined): Scene | null {
  return (palette && Object.hasOwn(scenes, palette) && scenes[palette as Palette]) || null
}
