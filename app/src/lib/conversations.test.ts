import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  contactTitle,
  conversationName,
  isGroupChat,
  isOutbound,
  messageCount,
  messageSource,
  receivedCount,
  reportTitle,
  sentCount,
  shortName,
  voicesOf,
} from './conversations.ts'

test('a saved name is the name a conversation goes by', () => {
  assert.equal(conversationName('Avery Stone', '+15550100199'), 'Avery Stone')
  assert.equal(conversationName('  Avery Stone  ', 'chat4471'), 'Avery Stone')
})

test('a conversation with no name and no id is unknown', () => {
  assert.equal(conversationName('', ''), 'Unknown conversation')
  assert.equal(conversationName(null, null), 'Unknown conversation')
  assert.equal(conversationName(undefined, '   '), 'Unknown conversation')
})

test('a group nobody named is an unnamed group', () => {
  assert.equal(conversationName('', 'chat4471'), 'Unnamed group')
  assert.equal(conversationName('   ', 'chat880011223344'), 'Unnamed group')
})

test('an address that starts with the word chat is a person, not a group', () => {
  assert.equal(conversationName('', 'chatterbox@example.com'), 'chatterbox@example.com')
  assert.equal(isGroupChat('chatterbox@example.com'), false)
  assert.equal(isGroupChat('chat'), false)
  assert.equal(isGroupChat('chat12a'), false)
})

test('a group id is the word chat and then digits only', () => {
  assert.equal(isGroupChat('chat4471'), true)
  assert.equal(isGroupChat(' chat4471 '), true)
  assert.equal(isGroupChat('+15550100199'), false)
  assert.equal(isGroupChat(''), false)
  assert.equal(isGroupChat(null), false)
  assert.equal(isGroupChat(undefined), false)
})

test('an email address is its own name', () => {
  assert.equal(conversationName('', 'avery@example.com'), 'avery@example.com')
})

test('a short code is its own name', () => {
  assert.equal(conversationName('', '55501'), '55501')
  assert.equal(conversationName('', '555-012'), '555-012')
})

test('a phone number is never shown in full', () => {
  assert.equal(conversationName('', '+15550100199'), 'Number ending in 0199')
  assert.equal(conversationName('', '(555) 010-0142'), 'Number ending in 0142')
  assert.equal(conversationName('', '5550142'), 'Number ending in 0142')
})

test('a saved person is called by their first name in a sentence', () => {
  assert.equal(shortName('Avery Stone'), 'Avery')
  assert.equal(shortName('  Avery   Stone  '), 'Avery')
  assert.equal(shortName('Avery'), 'Avery')
  assert.equal(shortName(''), '')
})

test('a label that stands in for a name is never shortened', () => {
  // every label conversationName can make
  const labels = [
    conversationName('', ''),
    conversationName('', 'chat4471'),
    conversationName('', 'avery@example.com'),
    conversationName('', '55501'),
    conversationName('', '+15550100199'),
  ]

  labels.forEach((label) => assert.equal(shortName(label), label))
  // the people of a group nobody named
  assert.equal(shortName('Avery Stone and Blake Reyes'), 'Avery Stone and Blake Reyes')
  assert.equal(shortName('Avery Stone, Blake Reyes and 2 more'), 'Avery Stone, Blake Reyes and 2 more')
  assert.equal(shortName('Avery & Blake'), 'Avery & Blake')
})

test('a contact and a report go by the same name', () => {
  assert.equal(contactTitle({ chat_id: '+15550100199', display_name: 'Avery Stone' }), 'Avery Stone')
  assert.equal(contactTitle({ chat_id: '+15550100199', displayName: 'Avery' }), 'Avery')
  assert.equal(contactTitle({ chat_id: '+15550100199' }), 'Number ending in 0199')

  const report = { name: 'analysis_15550100199.md', path: '/reports/analysis_15550100199.md', kind: 'md' }
  assert.equal(reportTitle({ ...report, contact: '+15550100199', displayName: 'Avery Stone' }), 'Avery Stone')
  assert.equal(reportTitle({ ...report, contact: '+15550100199' }), 'Number ending in 0199')
  assert.equal(reportTitle({ ...report, contact: '' }), 'analysis_15550100199.md')
})

test('a conversation never says it sent more than it holds', () => {
  assert.equal(messageCount({ chat_id: 'a', message_count: 12 }), 12)
  assert.equal(messageCount({ chat_id: 'a' }), 0)
  assert.equal(sentCount({ chat_id: 'a', message_count: 12, sent_count: 5 }), 5)
  assert.equal(sentCount({ chat_id: 'a', message_count: 12, sent_count: 40 }), 12)
})

test('what each side wrote is what the list says, not what is left of the total', () => {
  // three of the ten do not say who sent them: they are in the total and in neither voice
  const contact = { chat_id: 'a', message_count: 10, sent_count: 4, received_count: 3 }

  assert.deepEqual(voicesOf(contact), { sent: 4, received: 3 })
  assert.equal(receivedCount(contact), 3)
})

test('a list that does not say who wrote what gives no count for either voice', () => {
  // the Messages database is listed with totals only
  assert.equal(voicesOf({ chat_id: 'a', message_count: 10 }), null)
  assert.equal(voicesOf({ chat_id: 'a', message_count: 10, sent_count: 4 }), null)
  // and what was received is not made up from the total
  assert.equal(receivedCount({ chat_id: 'a', message_count: 10 }), 0)
  assert.equal(receivedCount({ chat_id: 'a', message_count: 10, sent_count: 4 }), 0)
})

test('the two voices together are never more than the conversation holds', () => {
  const within = (sent: number, received: number) =>
    voicesOf({ chat_id: 'a', message_count: 12, sent_count: sent, received_count: received })

  assert.deepEqual(within(5, 40), { sent: 5, received: 7 })
  assert.deepEqual(within(40, 40), { sent: 12, received: 0 })
  assert.deepEqual(within(5, -3), { sent: 5, received: 0 })
})

test('a message is yours when the server says so in any of its ways', () => {
  const yours: unknown[] = [true, 1, '1', 'true', 'True', 'TRUE', ' true ']
  const notYours: unknown[] = [false, 0, '0', 'false', '', 'yes please', 2, null, undefined]

  yours.forEach((value) => assert.equal(isOutbound({ isFromMe: value as boolean }), true, `yours: ${String(value)}`))
  notYours.forEach((value) => assert.equal(isOutbound({ isFromMe: value as boolean }), false, `not yours: ${String(value)}`))
  assert.equal(isOutbound({}), false)
})

const hit = (fields: Record<string, unknown>) => ({
  chatId: '+15550100199',
  timestamp: '2025-03-20 09:00:00',
  text: 'on my way',
  ...fields,
})

test('a found message says who it is from and where', () => {
  assert.equal(messageSource(hit({ displayName: 'Avery Stone', isFromMe: true })), 'You to Avery Stone')
  assert.equal(messageSource(hit({ displayName: 'Avery Stone', isFromMe: false })), 'Avery Stone')
  assert.equal(messageSource(hit({ isFromMe: false })), 'Number ending in 0199')
})

test('a found message in a group names its sender', () => {
  const inGroup = (fields: Record<string, unknown>) => hit({ chatId: 'chat4471', displayName: 'Trip', ...fields })

  assert.equal(messageSource(inGroup({ isFromMe: true })), 'You in Trip')
  const saved = inGroup({ isFromMe: false, senderName: 'Blake Reyes', sender: '+15550100142' })

  assert.equal(messageSource(saved), 'Blake Reyes in Trip')
  // someone who is not in Contacts is known by the end of their number
  assert.equal(messageSource(inGroup({ isFromMe: false, sender: '+15550100142' })), 'Number ending in 0142 in Trip')
  assert.equal(messageSource(inGroup({ isFromMe: false })), 'Trip')
  assert.equal(messageSource(hit({ chatId: 'chat4471', isFromMe: false })), 'Unnamed group')
})

test('a message from an address that starts with the word chat is from a person', () => {
  assert.equal(messageSource(hit({ chatId: 'chatterbox@example.com', isFromMe: true })), 'You to chatterbox@example.com')
})
