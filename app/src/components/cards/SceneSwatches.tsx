import { useLayoutEffect, useRef, type KeyboardEvent } from 'react'
import { SCENES, paintScene, type SceneName } from '../../lib/cardScenes'

// a swatch is the scene itself, drawn small: 14 by 9 cells of 4px
const COLS = 14
const ROWS = 9
const CELL = 4

function Swatch({ scene }: { scene: SceneName }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useLayoutEffect(() => {
    const pen = canvas.current?.getContext('2d')
    if (pen) paintScene(pen, scene, COLS, ROWS, CELL)
  }, [scene])
  return <canvas ref={canvas} width={COLS * CELL} height={ROWS * CELL} aria-hidden="true" />
}

// The scenes a card can stand on, as small pictures of themselves, after the
// swatches under Spotify's share card. One stop for Tab; the arrow keys
// choose, as in a set of radio buttons.
export function SceneSwatches({ value, onChange }: { value: SceneName; onChange: (scene: SceneName) => void }) {
  const group = useRef<HTMLDivElement>(null)
  const at = SCENES.findIndex((scene) => scene.value === value)

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const steps: Record<string, number> = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }
    if (!(event.key in steps)) return
    event.preventDefault()
    const next = (at + steps[event.key] + SCENES.length) % SCENES.length
    onChange(SCENES[next].value)
    group.current?.querySelectorAll<HTMLButtonElement>('[role="radio"]')[next]?.focus()
  }

  return (
    <div ref={group} className="scene-swatches" role="radiogroup" aria-label="Scene" onKeyDown={handleKeyDown}>
      {SCENES.map((scene) => {
        const chosen = scene.value === value
        return (
          <button
            key={scene.value}
            type="button"
            role="radio"
            aria-checked={chosen}
            aria-label={scene.label}
            title={scene.label}
            tabIndex={chosen ? 0 : -1}
            className="scene-swatch"
            onClick={() => onChange(scene.value)}
          >
            <Swatch scene={scene.value} />
          </button>
        )
      })}
    </div>
  )
}
