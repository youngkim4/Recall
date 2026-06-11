// Share-card rendering, straight onto a canvas 2D context. No DOM
// serialization, no foreignObject, no font embedding: the page's loaded
// Fraunces/Hanken faces are available to canvas after document.fonts.ready,
// and the same draw function feeds both the live preview and the PNG export
// so they can never drift.

export type CardFormat = 'story' | 'square'

export type AnniversaryCardData = {
  kind: 'anniversary'
  name: string
  years: number
  sinceDate: string
  messageCount: number
  quote?: { text: string; direction: string } | null
}

export type OnThisDayCardData = {
  kind: 'onthisday'
  name: string
  year: number
  yearsAgo: number
  count: number
  quote: string
}

export type DuoCardData = {
  kind: 'duo'
  name: string
  totalMessages: number
  initiation?: number | null
  balance?: number | null
  volumeTrendPct?: number | null
  busiestDay?: string
  busiestDayCount?: number | null
}

export type CardData = AnniversaryCardData | OnThisDayCardData | DuoCardData

export type CardOptions = {
  format: CardFormat
  includeQuote: boolean
}

export function cardSize(format: CardFormat): { w: number; h: number } {
  return format === 'story' ? { w: 1080, h: 1920 } : { w: 1080, h: 1080 }
}

type Palette = {
  bg: string
  surface: string
  text: string
  muted: string
  soft: string
  accent: string
  line: string
  lineStrong: string
}

function readPalette(): Palette {
  const styles = getComputedStyle(document.documentElement)
  const token = (name: string, fallback: string) => styles.getPropertyValue(name).trim() || fallback
  return {
    bg: token('--bg', '#f4eee2'),
    surface: token('--surface', '#fbf6ec'),
    text: token('--text', '#2a2017'),
    muted: token('--muted', '#6e6253'),
    soft: token('--soft', '#9a8d78'),
    accent: token('--accent', 'rgb(193, 60, 39)'),
    line: token('--line', 'rgba(43, 33, 24, 0.14)'),
    lineStrong: token('--line-strong', 'rgba(43, 33, 24, 0.28)'),
  }
}

const SERIF = 'Fraunces, Georgia, serif'
const SANS = '"Hanken Grotesk", -apple-system, sans-serif'

function breakLongWord(ctx: CanvasRenderingContext2D, word: string, maxWidth: number): string[] {
  if (ctx.measureText(word).width <= maxWidth) return [word]
  const parts: string[] = []
  let current = ''
  for (const char of word) {
    if (current && ctx.measureText(current + char).width > maxWidth) {
      parts.push(current)
      current = char
    } else {
      current += char
    }
  }
  if (current) parts.push(current)
  return parts
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  maxLines: number,
): string[] {
  // texting produces unbroken keyboard-smash tokens wider than the card --
  // hard-split those so a quote can never overflow the frame
  const words = text
    .split(/\s+/)
    .filter(Boolean)
    .flatMap((word) => breakLongWord(ctx, word, maxWidth))
  const lines: string[] = []
  let current = ''
  for (const word of words) {
    const attempt = current ? `${current} ${word}` : word
    if (ctx.measureText(attempt).width <= maxWidth || !current) {
      current = attempt
    } else {
      lines.push(current)
      current = word
      if (lines.length === maxLines - 1) break
    }
  }
  if (current && lines.length < maxLines) lines.push(current)
  const used = lines.join(' ').split(/\s+/).length
  if (used < words.length && lines.length) {
    lines[lines.length - 1] = `${lines[lines.length - 1].replace(/[.,!?]*$/, '')}…`
  }
  return lines
}

function kicker(ctx: CanvasRenderingContext2D, p: Palette, text: string, cx: number, y: number) {
  ctx.fillStyle = p.accent
  ctx.font = `600 30px ${SANS}`
  ctx.textAlign = 'center'
  const spaced = text.toUpperCase().split('').join('  ')
  ctx.fillText(spaced, cx, y)
}

function wordmark(ctx: CanvasRenderingContext2D, p: Palette, cx: number, bottomY: number) {
  ctx.strokeStyle = p.lineStrong
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(cx - 70, bottomY - 64)
  ctx.lineTo(cx + 70, bottomY - 64)
  ctx.stroke()
  ctx.fillStyle = p.text
  ctx.font = `700 40px ${SERIF}`
  ctx.textAlign = 'center'
  ctx.fillText('Recall', cx, bottomY)
}

function frame(ctx: CanvasRenderingContext2D, p: Palette, w: number, h: number) {
  ctx.fillStyle = p.bg
  ctx.fillRect(0, 0, w, h)
  ctx.strokeStyle = p.lineStrong
  ctx.lineWidth = 2
  ctx.strokeRect(50, 50, w - 100, h - 100)
  ctx.strokeStyle = p.line
  ctx.lineWidth = 1
  ctx.strokeRect(62, 62, w - 124, h - 124)
}

function firstName(name: string): string {
  const first = name.trim().split(/\s+/)[0] || name
  return first
}

export function drawCard(ctx: CanvasRenderingContext2D, data: CardData, opts: CardOptions) {
  const { w, h } = cardSize(opts.format)
  const p = readPalette()
  const cx = w / 2
  frame(ctx, p, w, h)

  if (data.kind === 'anniversary') {
    drawAnniversary(ctx, p, data, opts, w, h, cx)
  } else if (data.kind === 'onthisday') {
    drawOnThisDay(ctx, p, data, opts, w, h, cx)
  } else {
    drawDuo(ctx, p, data, w, h, cx)
  }

  wordmark(ctx, p, cx, h - 110)
}

function drawAnniversary(
  ctx: CanvasRenderingContext2D,
  p: Palette,
  data: AnniversaryCardData,
  opts: CardOptions,
  w: number,
  h: number,
  cx: number,
) {
  const story = h > w
  const topY = story ? 360 : 220
  kicker(ctx, p, 'Friendship anniversary', cx, topY)

  const numeralY = topY + (story ? 360 : 290)
  ctx.fillStyle = p.text
  ctx.font = `900 ${story ? 430 : 330}px ${SERIF}`
  ctx.textAlign = 'center'
  ctx.fillText(String(data.years), cx, numeralY)

  ctx.font = `italic 500 ${story ? 60 : 52}px ${SERIF}`
  ctx.fillStyle = p.text
  ctx.fillText(
    `${data.years === 1 ? 'year' : 'years'} of you & ${firstName(data.name)}`,
    cx,
    numeralY + (story ? 110 : 90),
  )

  ctx.font = `500 ${story ? 36 : 32}px ${SANS}`
  ctx.fillStyle = p.muted
  ctx.fillText(
    `since ${data.sinceDate} · ${data.messageCount.toLocaleString()} messages`,
    cx,
    numeralY + (story ? 180 : 150),
  )

  if (opts.includeQuote && data.quote?.text) {
    const quoteY = numeralY + (story ? 330 : 250)
    ctx.font = `italic 500 ${story ? 46 : 40}px ${SERIF}`
    ctx.fillStyle = p.text
    const lines = wrapText(ctx, `“${data.quote.text}”`, w - 320, 3)
    lines.forEach((line, index) => {
      ctx.fillText(line, cx, quoteY + index * (story ? 62 : 54))
    })
    ctx.font = `500 30px ${SANS}`
    ctx.fillStyle = p.soft
    const who = data.quote.direction === 'outgoing' ? 'your first words' : 'their first words'
    ctx.fillText(who, cx, quoteY + lines.length * (story ? 62 : 54) + 34)
  }
}

function drawOnThisDay(
  ctx: CanvasRenderingContext2D,
  p: Palette,
  data: OnThisDayCardData,
  opts: CardOptions,
  w: number,
  h: number,
  cx: number,
) {
  const story = h > w
  kicker(ctx, p, `On this day · ${data.year}`, cx, story ? 420 : 240)

  const quoteText = opts.includeQuote && data.quote ? data.quote : ''
  const centerY = story ? 850 : 480
  if (quoteText) {
    ctx.font = `italic 500 ${story ? 58 : 50}px ${SERIF}`
    ctx.fillStyle = p.text
    const lines = wrapText(ctx, `“${quoteText}”`, w - 300, story ? 6 : 4)
    const lineHeight = story ? 80 : 68
    const startY = centerY - ((lines.length - 1) * lineHeight) / 2
    lines.forEach((line, index) => {
      ctx.fillText(line, cx, startY + index * lineHeight)
    })
  } else {
    ctx.font = `600 ${story ? 130 : 110}px ${SERIF}`
    ctx.fillStyle = p.text
    ctx.fillText(data.count.toLocaleString(), cx, centerY - 20)
    ctx.font = `italic 500 ${story ? 52 : 44}px ${SERIF}`
    ctx.fillText('messages, this day', cx, centerY + (story ? 70 : 56))
  }

  const attributionY = story ? h - 480 : h - 330
  ctx.font = `500 ${story ? 38 : 34}px ${SANS}`
  ctx.fillStyle = p.muted
  ctx.fillText(
    `— ${firstName(data.name)}, ${data.yearsAgo} ${data.yearsAgo === 1 ? 'year' : 'years'} ago today`,
    cx,
    attributionY,
  )
  ctx.font = `500 30px ${SANS}`
  ctx.fillStyle = p.soft
  ctx.fillText(`${data.count.toLocaleString()} messages that day`, cx, attributionY + 52)
}

function drawDuo(
  ctx: CanvasRenderingContext2D,
  p: Palette,
  data: DuoCardData,
  w: number,
  h: number,
  cx: number,
) {
  const story = h > w
  kicker(ctx, p, `You & ${firstName(data.name)}`, cx, story ? 380 : 220)

  const totalY = story ? 660 : 420
  ctx.fillStyle = p.text
  ctx.font = `600 ${story ? 170 : 140}px ${SERIF}`
  ctx.textAlign = 'center'
  ctx.fillText(data.totalMessages.toLocaleString(), cx, totalY)
  ctx.font = `italic 500 ${story ? 52 : 44}px ${SERIF}`
  ctx.fillText('messages together', cx, totalY + (story ? 86 : 70))

  const rows: Array<[string, string]> = []
  if (data.initiation !== null && data.initiation !== undefined) {
    rows.push(['You start the day', `${Math.round(data.initiation * 100)}%`])
  }
  if (data.balance !== null && data.balance !== undefined) {
    rows.push(['Share of words', `${Math.round(data.balance * 100)}% you`])
  }
  if (data.volumeTrendPct !== null && data.volumeTrendPct !== undefined) {
    const sign = data.volumeTrendPct >= 0 ? '+' : ''
    rows.push(['Lately', `${sign}${Math.round(data.volumeTrendPct)}%`])
  }
  if (data.busiestDay) {
    rows.push([
      'Biggest day',
      `${data.busiestDay}${data.busiestDayCount ? ` · ${data.busiestDayCount.toLocaleString()} texts` : ''}`,
    ])
  }

  const rowsTop = totalY + (story ? 220 : 160)
  const rowHeight = story ? 110 : 92
  const left = 180
  const right = w - 180
  rows.slice(0, story ? 4 : 3).forEach(([label, value], index) => {
    const y = rowsTop + index * rowHeight
    ctx.strokeStyle = p.line
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(left, y - rowHeight / 2 + 14)
    ctx.lineTo(right, y - rowHeight / 2 + 14)
    ctx.stroke()
    ctx.font = `500 ${story ? 36 : 32}px ${SANS}`
    ctx.fillStyle = p.muted
    ctx.textAlign = 'left'
    ctx.fillText(label, left, y + 12)
    ctx.font = `600 ${story ? 40 : 36}px ${SERIF}`
    ctx.fillStyle = p.text
    ctx.textAlign = 'right'
    ctx.fillText(value, right, y + 12)
  })
  ctx.textAlign = 'center'
}

export async function renderCardToCanvas(
  canvas: HTMLCanvasElement,
  data: CardData,
  opts: CardOptions,
): Promise<void> {
  const { w, h } = cardSize(opts.format)
  canvas.width = w
  canvas.height = h
  try {
    await document.fonts.ready
  } catch {
    // system serif fallback still renders a correct card
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, w, h)
  drawCard(ctx, data, opts)
}

export async function exportCardPng(data: CardData, opts: CardOptions, filename: string) {
  const canvas = document.createElement('canvas')
  await renderCardToCanvas(canvas, data, opts)
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  if (!blob) return
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 10000)
}
