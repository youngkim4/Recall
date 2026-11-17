// The story of a report, taken apart so it can be set as chapters.
//
// The server writes the story's HTML from an allowlist (story_html in
// safe_html.py): a tag carries no attribute but a link's address and a list's
// start, and every "<" that was text is escaped. So a tag found here is a real
// tag. The pieces cut below are whole elements of that HTML with only the
// level of a heading changed; no text is ever turned into markup.

export type StoryChapter = {
  // a place on the page to jump to
  id: string
  // the heading in plain words
  title: string
  // what stands under the heading
  html: string
}

export type Story = {
  // the story's own title, when it starts with a heading that stands over all of it; else ''
  title: string
  // what comes before the first chapter
  opening: string
  chapters: StoryChapter[]
  // how long it takes to read, in whole minutes
  minutes: number
}

type Block = { tag: string; outer: string; inner: string }
type Scan = { blocks: Block[]; depth: number; start: number; innerStart: number; end: number }

const TAG = /<(\/?)([a-z][a-z0-9]*)(?:\s[^>]*)?>/g
const VOID = new Set(['br', 'hr'])
const HEADING = /^h([1-6])$/
const ESCAPES: Record<string, string> = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#x27;': "'" }
const WORDS_A_MINUTE = 230

// the text of a piece of the story: tags dropped, the five escapes read once
export function plainText(html: string) {
  return html.replace(/<[^>]*>/g, '').replace(/&(?:amp|lt|gt|quot|#x27);/g, (escape) => ESCAPES[escape])
}

// loose text between two elements, as a block of its own
const loose = (text: string): Block[] => (text.trim() ? [{ tag: '', outer: text, inner: text }] : [])

// The elements that stand at the top of the HTML, in order. What is inside
// one of them, a heading in a list for one, is part of it and not a block.
function topBlocks(html: string): Block[] {
  const scan = [...html.matchAll(TAG)].reduce<Scan>(
    (state, match) => {
      const [whole, slash, tag] = match
      const at = match.index
      if (VOID.has(tag)) {
        if (state.depth > 0) return state
        return {
          ...state,
          blocks: [...state.blocks, ...loose(html.slice(state.end, at)), { tag, outer: whole, inner: '' }],
          end: at + whole.length,
        }
      }
      if (!slash) {
        if (state.depth > 0) return { ...state, depth: state.depth + 1 }
        return {
          ...state,
          blocks: [...state.blocks, ...loose(html.slice(state.end, at))],
          depth: 1,
          start: at,
          innerStart: at + whole.length,
        }
      }
      if (state.depth !== 1) return { ...state, depth: Math.max(0, state.depth - 1) }
      const end = at + whole.length
      const block = { tag, outer: html.slice(state.start, end), inner: html.slice(state.innerStart, at) }
      return { ...state, blocks: [...state.blocks, block], depth: 0, end }
    },
    { blocks: [], depth: 0, start: 0, innerStart: 0, end: 0 },
  )
  // an element that was never closed keeps what it holds
  const rest = scan.depth > 0 ? [{ tag: '', outer: html.slice(scan.start), inner: '' }] : loose(html.slice(scan.end))
  return [...scan.blocks, ...rest]
}

const levelOf = (block: Block) => Number(HEADING.exec(block.tag)?.[1] ?? 0)

// The level the story is cut at: the highest one that has two headings or
// more, so a single title over everything does not make one chapter of it all.
function cutLevel(blocks: Block[]) {
  const levels = blocks.map(levelOf).filter(Boolean)
  const repeated = [1, 2, 3, 4, 5, 6].find((level) => levels.filter((found) => found === level).length > 1)
  return repeated ?? (levels.length ? Math.min(...levels) : 0)
}

// the level the page sets a chapter's own heading at
const CHAPTER_LEVEL = 4

// The headings under a chapter step down from the chapter's in order; one at
// or above it, the lone title of a story, is set as a chapter heading too.
function underChapter(html: string, cut: number) {
  return html.replace(/<(\/?)h([1-6])>/g, (_tag, slash: string, level: string) => {
    const below = Math.max(0, Number(level) - cut)
    return `<${slash}h${Math.min(6, CHAPTER_LEVEL + below)}>`
  })
}

const joined = (blocks: Block[], cut: number) => underChapter(blocks.map((block) => block.outer).join(''), cut)

export function readStory(html: string): Story {
  const blocks = topBlocks(html || '')
  if (!blocks.length) return { title: '', opening: '', chapters: [], minutes: 0 }
  const cut = cutLevel(blocks)
  const firstChapter = blocks.findIndex((block) => cut > 0 && levelOf(block) === cut)
  const before = firstChapter < 0 ? blocks : blocks.slice(0, firstChapter)
  // a heading above the chapters that the story starts with stands over all of it
  const titled = before.length > 0 && levelOf(before[0]) > 0
  const chapters = (firstChapter < 0 ? [] : blocks.slice(firstChapter)).reduce<Array<{ title: string; blocks: Block[] }>>(
    (parts, block) => {
      if (levelOf(block) === cut) return [...parts, { title: plainText(block.inner).trim(), blocks: [] }]
      const open = parts[parts.length - 1]
      return [...parts.slice(0, -1), { ...open, blocks: [...open.blocks, block] }]
    },
    [],
  )
  const words = plainText(html).split(/\s+/).filter(Boolean).length
  return {
    title: titled ? plainText(before[0].inner).trim() : '',
    opening: joined(titled ? before.slice(1) : before, cut),
    chapters: chapters.map((part, index) => ({
      id: `chapter-${index + 1}`,
      title: part.title,
      html: joined(part.blocks, cut),
    })),
    minutes: Math.max(1, Math.ceil(words / WORDS_A_MINUTE)),
  }
}
