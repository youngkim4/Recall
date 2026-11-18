import { useEffect, useId, useRef } from 'react'
import { frameOf, type Scene } from '../../lib/backdrop/craft'
import { STILL_AT } from '../../lib/backdrop/run'
import { sceneOf } from '../../lib/backdrop/scenes'
import { paletteOptions, type Palette } from '../../lib/palette'
import { usePageIsDark } from '../../lib/usePage'

type PalettePickerProps = {
  value: Palette
  onChange: (palette: Palette) => void
  // the id of the words that name the setting
  labelledBy: string
}

// the small picture of a scene is the rail of the card: cells of 2px, rows in its top quarter and at its foot
const SMALL = { cell: 2, nav: 0.25, foot: 0.86 }

// one still frame of a scene, for the card of its palette
function ScenePicture({ scene, dark }: { scene: Scene; dark: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const element = canvas.current
    const context = element?.getContext('2d')
    if (!element || !context) return
    // as many cells as the card shows, so the picture is not stretched
    const cols = Math.max(1, Math.round(element.clientWidth / SMALL.cell))
    const rows = Math.max(1, Math.round(element.clientHeight / SMALL.cell))
    element.width = cols
    element.height = rows
    const frame = frameOf(cols, rows, cols, Math.round(rows * SMALL.nav), Math.round(rows * SMALL.foot), 3)
    scene.paint(frame, STILL_AT, dark)
    context.putImageData(new ImageData(frame.cells, cols, rows), 0, 0)
  }, [scene, dark])
  return <canvas ref={canvas} className="palette-preview-scene" />
}

// The palettes as cards. Each card is a small picture of the app in that
// palette, in the light or dark the app is in now: the rail, the page, a
// message received and a message sent, over the palette's name. A palette
// that is a scene shows its picture behind them. They are the radio buttons
// of one group: the arrow keys go from one to the next.
export function PalettePicker({ value, onChange, labelledBy }: PalettePickerProps) {
  const name = useId()
  const dark = usePageIsDark()
  return (
    <div className="palette-picker" role="radiogroup" aria-labelledby={labelledBy}>
      {paletteOptions.map((option) => {
        const scene = sceneOf(option.value)
        return (
          <label key={option.value} className="palette-card">
            <input
              className="sr-only"
              type="radio"
              name={name}
              value={option.value}
              checked={option.value === value}
              onChange={() => onChange(option.value)}
            />
            {/* the picture carries the palette's name, so its colours are that palette's and not the page's */}
            <span className="palette-preview" data-palette={option.value} data-scene={scene ? '' : undefined} aria-hidden="true">
              {scene ? <ScenePicture scene={scene} dark={dark} /> : null}
              <span className="palette-preview-rail" />
              <span className="palette-preview-page">
                <span className="palette-preview-them" />
                <span className="palette-preview-you" />
              </span>
            </span>
            <span className="palette-name">{option.label}</span>
          </label>
        )
      })}
    </div>
  )
}
