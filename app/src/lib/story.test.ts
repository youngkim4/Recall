import assert from 'node:assert/strict'
import { test } from 'node:test'
import { plainText, readStory } from './story.ts'

test('a story is cut into chapters at its headings', () => {
  const story = readStory('<h3>How it began</h3><p>They met in spring.</p><h3>What changed</h3><p>Then it slowed.</p>')

  assert.deepEqual(
    story.chapters.map((chapter) => [chapter.title, chapter.html]),
    [
      ['How it began', '<p>They met in spring.</p>'],
      ['What changed', '<p>Then it slowed.</p>'],
    ],
  )
})

test('each chapter has a place of its own to jump to', () => {
  const story = readStory('<h3>One</h3><p>a</p><h3>Two</h3><p>b</p><h3>One</h3><p>c</p>')
  const ids = story.chapters.map((chapter) => chapter.id)

  assert.equal(new Set(ids).size, 3)
})

test('what comes before the first heading is the opening of the story, not a chapter', () => {
  const story = readStory('<p>An opening line.</p><h3>Then</h3><p>More.</p>')

  assert.equal(story.title, '')
  assert.equal(story.opening, '<p>An opening line.</p>')
  assert.deepEqual(
    story.chapters.map((chapter) => [chapter.title, chapter.html]),
    [['Then', '<p>More.</p>']],
  )
})

test('a story with no headings is all opening', () => {
  const story = readStory('<p>One.</p><p>Two.</p>')

  assert.equal(story.opening, '<p>One.</p><p>Two.</p>')
  assert.deepEqual(story.chapters, [])
})

test('the headings under a chapter step down from it in order', () => {
  const story = readStory('<h2>Arc</h2><h3>Phase one</h3><p>a</p><h4>A detail</h4><p>b</p><h2>Now</h2><p>c</p>')

  assert.equal(story.chapters[0].title, 'Arc')
  // the page sets a chapter's own heading as h4
  assert.equal(story.chapters[0].html, '<h5>Phase one</h5><p>a</p><h6>A detail</h6><p>b</p>')
})

test('a single heading over the whole story is its title, and the sections under it are the chapters', () => {
  const story = readStory('<h2>The whole thing</h2><p>Opening.</p><h3>First</h3><p>a</p><h3>Second</h3><p>b</p>')

  assert.equal(story.title, 'The whole thing')
  assert.equal(story.opening, '<p>Opening.</p>')
  assert.deepEqual(
    story.chapters.map((chapter) => chapter.title),
    ['First', 'Second'],
  )
})

test('a heading in the opening that is not at its very start stays in the opening', () => {
  const story = readStory('<p>First words.</p><h2>A heading</h2><p>More.</p><h3>One</h3><p>a</p><h3>Two</h3><p>b</p>')

  assert.equal(story.title, '')
  assert.equal(story.opening, '<p>First words.</p><h4>A heading</h4><p>More.</p>')
})

test('a heading inside a list or a quote does not cut the story', () => {
  const html = '<h3>One</h3><ul><li><h3>Not a chapter</h3><p>x</p></li></ul><blockquote><h3>Nor this</h3></blockquote>'
  const story = readStory(html)

  assert.equal(story.chapters.length, 1)
  assert.equal(story.chapters[0].html, '<ul><li><h4>Not a chapter</h4><p>x</p></li></ul><blockquote><h4>Nor this</h4></blockquote>')
})

test('a heading is given as plain words, its marks read and its tags dropped', () => {
  const story = readStory('<h3>Tom &amp; <em>Ana</em>: &quot;later&quot; &lt;3</h3><p>a</p><h3>b</h3><p>c</p>')

  assert.equal(story.chapters[0].title, 'Tom & Ana: "later" <3')
})

test('text that was escaped stays escaped in what is put on the page', () => {
  const story = readStory('<h3>A</h3><p>&lt;script&gt;alert(1)&lt;/script&gt;</p><h3>B</h3><p>b</p>')

  assert.equal(story.chapters[0].html, '<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>')
  assert.equal(story.chapters[0].html.includes('<script'), false)
})

test('a rule or a table between paragraphs stays where it was', () => {
  const html = '<h3>A</h3><p>a</p><hr><table><thead><tr><th>x</th></tr></thead></table><h3>B</h3><p>b</p>'
  const story = readStory(html)

  assert.equal(story.chapters[0].html, '<p>a</p><hr><table><thead><tr><th>x</th></tr></thead></table>')
})

test('a link keeps its address', () => {
  const story = readStory('<h3>A</h3><p>See <a href="https://example.com/a?b=1&amp;c=2">this</a>.</p><h3>B</h3><p>b</p>')

  assert.equal(story.chapters[0].html, '<p>See <a href="https://example.com/a?b=1&amp;c=2">this</a>.</p>')
})

test('how long a story takes to read is counted in whole minutes, never none', () => {
  const words = Array.from({ length: 700 }, () => 'word').join(' ')

  assert.equal(readStory(`<p>${words}</p>`).minutes, 4)
  assert.equal(readStory('<p>Short.</p>').minutes, 1)
})

test('no story is nothing at all', () => {
  const nothing = { title: '', opening: '', chapters: [], minutes: 0 }

  assert.deepEqual(readStory(''), nothing)
  assert.deepEqual(readStory('   \n '), nothing)
})

test('plain words are the text with its five escapes read once', () => {
  assert.equal(plainText('<p>a &amp;lt; b &#x27;c&#x27;</p>'), "a &lt; b 'c'")
})
