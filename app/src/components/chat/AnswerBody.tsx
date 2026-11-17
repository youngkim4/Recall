import type { ReactNode } from 'react'

const INLINE_PATTERN = /(\*\*[^*]+\*\*|\[\d+\])/g

// Inline formatter: **bold** and [n] citation markers. A marker attaches to
// the word before it with no space, and markers that sit side by side are
// separated by a comma so "[1][2]" cannot be read as twelve.
function renderInline(text: string, keyBase: string): ReactNode[] {
  const parts = text.replace(/\s+(\[\d+\])/g, '$1').split(INLINE_PATTERN)
  return parts.map((part, index) => {
    if (!part) return null
    const bold = /^\*\*([^*]+)\*\*$/.exec(part)
    if (bold) return <strong key={`${keyBase}-b${index}`}>{bold[1]}</strong>
    const cite = /^\[(\d+)\]$/.exec(part)
    if (cite) {
      // split() leaves an empty string between two adjacent markers
      const follows = index >= 2 && parts[index - 1] === '' && /^\[\d+\]$/.test(parts[index - 2])
      return (
        <sup key={`${keyBase}-c${index}`} className="cite-ref">
          {follows ? `,${cite[1]}` : cite[1]}
        </sup>
      )
    }
    return <span key={`${keyBase}-t${index}`}>{part}</span>
  })
}

// The answer as reading text: paragraphs, soft line breaks, bullet lists,
// inline bold and citation markers. While it streams, a caret follows the
// last character.
export function AnswerBody({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const blocks = text.trim().split(/\n{2,}/).filter(Boolean)
  const caret = streaming ? <span className="caret" aria-hidden="true" /> : null
  return (
    <div className="answer-body">
      {blocks.map((block, blockIndex) => {
        const lines = block.split('\n')
        const isLastBlock = blockIndex === blocks.length - 1
        const isList = lines.every((line) => /^\s*[-*]\s+/.test(line))
        if (isList) {
          return (
            <ul key={`block-${blockIndex}`}>
              {lines.map((line, lineIndex) => (
                <li key={`block-${blockIndex}-item-${lineIndex}`}>
                  {renderInline(line.replace(/^\s*[-*]\s+/, ''), `block-${blockIndex}-${lineIndex}`)}
                  {isLastBlock && lineIndex === lines.length - 1 ? caret : null}
                </li>
              ))}
            </ul>
          )
        }
        return (
          <p key={`block-${blockIndex}`}>
            {lines.map((line, lineIndex) => (
              <span key={`block-${blockIndex}-line-${lineIndex}`}>
                {renderInline(line, `block-${blockIndex}-${lineIndex}`)}
                {lineIndex < lines.length - 1 ? <br /> : null}
              </span>
            ))}
            {isLastBlock ? caret : null}
          </p>
        )
      })}
      {blocks.length === 0 ? caret : null}
    </div>
  )
}
