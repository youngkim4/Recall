// How the parts of a share card share its height. A card says more or less
// from one conversation to the next: a chart there is no data for, a message
// that was not asked onto it. The parts are placed afresh each time, so what
// is there always fills the card and nothing floats in a void.

import type { Pen } from './cardPen.ts'

export type Block = {
  // the least height it can be drawn in
  min: number
  // the most it can use; a part with none keeps to its least
  max?: number
  // 0 is always drawn; of the others, the highest is left off first when the card is too full
  rank?: number
  draw: (pen: Pen, top: number, height: number) => void
}

export type Placed = { block: Block; top: number; height: number }

const rankOf = (block: Block) => block.rank ?? 0
const needed = (blocks: Block[], gap: number) =>
  blocks.reduce((sum, block) => sum + block.min, 0) + gap * Math.max(0, blocks.length - 1)

// the parts that fit the room, the least important left off one at a time
function fitting(blocks: Block[], room: number, gap: number): Block[] {
  const worst = Math.max(0, ...blocks.map(rankOf))
  if (needed(blocks, gap) <= room || worst === 0) return blocks
  const dropped = blocks.findLastIndex((block) => rankOf(block) === worst)
  return fitting(
    blocks.filter((_, index) => index !== dropped),
    room,
    gap,
  )
}

// Places the parts between top and bottom, a gap apart, in their order. Room
// that is left goes first to the parts that can grow, and what remains is
// shared above and below them, so they sit together in the middle.
export function place(blocks: Block[], top: number, bottom: number, gap: number): Placed[] {
  const room = bottom - top
  const kept = fitting(blocks, room, gap)
  const spare = Math.max(0, room - needed(kept, gap))
  const grown = kept.reduce<{ heights: number[]; spare: number }>(
    (state, block) => {
      const more = Math.min(state.spare, Math.max(0, (block.max ?? block.min) - block.min))
      return { heights: [...state.heights, block.min + more], spare: state.spare - more }
    },
    { heights: [], spare },
  )
  const first = top + grown.spare / 2
  return kept.map((block, index) => ({
    block,
    top: first + grown.heights.slice(0, index).reduce((sum, height) => sum + height + gap, 0),
    height: grown.heights[index],
  }))
}
