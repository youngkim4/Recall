import { useEffect, useRef } from 'react'
import { runScene, type Room } from '../lib/backdrop/run'
import { sceneOf } from '../lib/backdrop/scenes'
import { usePageAttribute, usePageIsDark } from '../lib/usePage'

// The room of the picture is the rail, and where its rows stand. With no
// rail there is no room: nothing is painted.
function roomNow(): Room {
  const height = window.innerHeight
  const rail = document.querySelector('.rail')?.getBoundingClientRect()
  const nav = document.querySelector('.rail > .nav')?.getBoundingClientRect()
  const foot = document.querySelector('.rail-foot')?.getBoundingClientRect()
  if (!rail || !nav || !foot) return { width: 0, height, railRight: 0, navBottom: 0, footTop: height }
  return { width: rail.right, height, railRight: rail.right, navBottom: nav.bottom, footTop: foot.top }
}

// The picture of a palette that has one. It is painted in the rail, under
// its rows, and takes no pointer. The workspace beside it is a plain page.
export function Backdrop() {
  const scene = sceneOf(usePageAttribute('data-palette'))
  const dark = usePageIsDark()
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!scene || !canvas.current) return
    const running = runScene(canvas.current, scene, dark, roomNow)
    // The rail opens, folds, is dragged wider, or comes after the scene: the
    // scene is composed around it. The app's frame holds the rail, so a
    // change to either is seen here.
    const watcher = new ResizeObserver(running.refit)
    for (const part of document.querySelectorAll('.app, .rail')) watcher.observe(part)
    return () => {
      watcher.disconnect()
      running.stop()
    }
  }, [scene, dark])

  if (!scene) return null
  return <canvas ref={canvas} className="backdrop" aria-hidden="true" />
}
