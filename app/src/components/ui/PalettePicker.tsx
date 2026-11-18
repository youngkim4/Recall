import { useId } from 'react'
import { paletteOptions, type Palette } from '../../lib/palette'

type PalettePickerProps = {
  value: Palette
  onChange: (palette: Palette) => void
  // the id of the words that name the setting
  labelledBy: string
}

// The palettes as a row of small rounds. Each shows its own ground and its
// own hue, whatever palette the page is in. They are the radio buttons of
// one group: the arrow keys go from one to the next.
export function PalettePicker({ value, onChange, labelledBy }: PalettePickerProps) {
  const name = useId()
  return (
    <div className="palette-picker" role="radiogroup" aria-labelledby={labelledBy}>
      {paletteOptions.map((option) => (
        <label key={option.value} className="palette-choice" title={option.label}>
          <input
            className="sr-only"
            type="radio"
            name={name}
            value={option.value}
            checked={option.value === value}
            onChange={() => onChange(option.value)}
          />
          <span className="sr-only">{option.label}</span>
          <span className="palette-swatch" data-palette={option.value} aria-hidden="true" />
        </label>
      ))}
    </div>
  )
}
