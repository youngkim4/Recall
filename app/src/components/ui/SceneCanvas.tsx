import { useLayoutEffect, useRef } from 'react'
import { paintScene, type SceneName } from '../../lib/cardScenes'

// cells of 8px: the coarse grid of the rail's scenes, at the size of a window
const CELL = 8
// the northern lights hang high, over the words in the middle of the window
const CURTAIN = 0.16

// One of Recall's scenes, filling the box it is put in and drawn again when
// the box changes size. It is a picture behind words, so it says nothing to a
// screen reader.
export function SceneCanvas({ scene, className }: { scene: SceneName; className?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useLayoutEffect(() => {
    const element = canvas.current
    if (!element) return
    const draw = () => {
      const { width, height } = element.getBoundingClientRect()
      const cols = Math.max(1, Math.ceil(width / CELL))
      const rows = Math.max(1, Math.ceil(height / CELL))
      element.width = cols * CELL
      element.height = rows * CELL
      const pen = element.getContext('2d')
      if (pen) paintScene(pen, scene, cols, rows, CELL, { x: 0, y: 0 }, CURTAIN)
    }
    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(element)
    return () => observer.disconnect()
  }, [scene])

  return <canvas ref={canvas} className={className} aria-hidden="true" />
}
