import { useEffect, useRef, useState } from 'react'
import { easeOut, mix, resample } from './tween.ts'

// how long a chart takes to go from one answer to the next
export const TWEEN_MS = 420

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

// The values of `target` as they are drawn: when the target changes, they go
// there from wherever they are, a frame at a time, fast at first and settling.
// A change in the middle of one goes on from where that one is. With reduced
// motion they jump. The first values are drawn as they are.
export function useTweened(target: readonly number[], ms = TWEEN_MS): readonly number[] {
  const [shown, setShown] = useState(target)
  // what is on the screen now, for the next change to start from
  const now = useRef(target)
  // the values as one string: a new array of the same numbers is no change
  const key = target.join(' ')

  useEffect(() => {
    const goal = key ? key.split(' ').map(Number) : []
    const from = resample(now.current, goal.length)
    const length = reducedMotion() ? 0 : ms
    const start = performance.now()
    let frame = 0
    const step = (time: number) => {
      const share = length ? (time - start) / length : 1
      const next = share >= 1 ? goal : mix(from, goal, easeOut(share))
      now.current = next
      setShown(next)
      if (share < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [key, ms])

  return shown
}
