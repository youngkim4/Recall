// The geometry of a sheet that grows out of what opened it: where the sheet
// starts, as a transform of its own box, and how much of it is cut away so
// that what is left has the size of the origin. Pure numbers; sheetZoom.ts
// draws them.

export type Box = { x: number; y: number; width: number; height: number }

// The sheet at the start: moved by (x, y), shrunk by `scale` from its top
// left corner, and cut by `right` and `bottom`, in the sheet's own pixels.
// The sheet shrinks until it covers the origin on one side and is cut to it
// on the other, so its words keep their shape and are never stretched.
export type ZoomFrame = { x: number; y: number; scale: number; right: number; bottom: number }

export function zoomFrame(origin: Box, sheet: Box): ZoomFrame | null {
  if (origin.width <= 0 || origin.height <= 0 || sheet.width <= 0 || sheet.height <= 0) return null
  const scale = Math.max(origin.width / sheet.width, origin.height / sheet.height)
  return {
    x: origin.x - sheet.x,
    y: origin.y - sheet.y,
    scale,
    right: sheet.width - origin.width / scale,
    bottom: sheet.height - origin.height / scale,
  }
}

type ZoomKeyframe = { transform: string; clipPath: string }

const shape = (x: number, y: number, scale: number, right: number, bottom: number, radius: number): ZoomKeyframe => ({
  transform: `translate(${x}px, ${y}px) scale(${scale})`,
  clipPath: `inset(0px ${right}px ${bottom}px 0px round ${radius}px)`,
})

// From the origin to the sheet in place. The corner is kept at the origin's
// radius on screen: a sheet at half size needs twice the radius.
export function zoomKeyframes(
  frame: ZoomFrame,
  originRadius: number,
  sheetRadius: number,
): [ZoomKeyframe, ZoomKeyframe] {
  return [
    shape(frame.x, frame.y, frame.scale, frame.right, frame.bottom, originRadius / frame.scale),
    shape(0, 0, 1, 0, 0, sheetRadius),
  ]
}
