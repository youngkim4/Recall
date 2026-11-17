import { useCallback, useEffect, useState } from 'react'
import type { PreviewPayload } from '../types'
import { recallApi } from './api'
import { isPreviewPayload, makePreviewCacheKey, readPreviewCache, writePreviewCache } from './cache'

type PreviewRequest = {
  // '' for any of the first three means there is nothing to ask for yet
  messagesPath: string
  contact: string
  model: string
  since: string
  until: string
}

type Held = {
  // the request the figures answer
  key: string
  // the archive and conversation they belong to
  subject: string
  payload: PreviewPayload | null
}

// the request that could not be answered, and why
type Failure = { key: string; message: string }

export type ConversationPreview = {
  // The figures for the conversation. While new dates or another quality are
  // worked out, the figures before them stay up, so the page never blanks
  // between two views of one conversation. They also stay when that fails.
  payload: PreviewPayload | null
  // true when the figures are the ones that were asked for
  settled: boolean
  // why the figures that were asked for are not here
  error: string | null
  retry: () => void
}

const NOTHING: Held = { key: '', subject: '', payload: null }

// Loads the figures for one conversation, from the browser's store first and
// then from the server. onStored is called when a fresh answer has been kept.
export function useConversationPreview(
  { messagesPath, contact, model, since, until }: PreviewRequest,
  onStored: () => void,
): ConversationPreview {
  const [held, setHeld] = useState<Held>(NOTHING)
  const [failure, setFailure] = useState<Failure | null>(null)
  const [attempt, setAttempt] = useState(0)
  const ready = Boolean(messagesPath && contact && model)
  const key = ready ? makePreviewCacheKey({ messagesPath, contact, model, since, until }) : ''
  const subject = ready ? `${messagesPath}\n${contact}` : ''

  useEffect(() => {
    if (!key) return
    let cancelled = false
    const stored = readPreviewCache(key)
    if (stored) {
      queueMicrotask(() => {
        if (!cancelled) setHeld({ key, subject, payload: stored })
      })
    }
    recallApi
      .preview({ messagesPath, contact, model, since, until })
      .then((payload) => {
        if (!isPreviewPayload(payload)) throw new Error('This conversation could not be read.')
        writePreviewCache(key, payload)
        if (cancelled) return
        onStored()
        setHeld({ key, subject, payload })
        setFailure(null)
      })
      .catch((apiError: unknown) => {
        // an answer from the store stays up when the server cannot be reached
        if (cancelled || stored) return
        setFailure({
          key,
          message: apiError instanceof Error ? apiError.message : 'This conversation could not be read.',
        })
      })
    return () => {
      cancelled = true
    }
  }, [key, subject, messagesPath, contact, model, since, until, onStored, attempt])

  const retry = useCallback(() => {
    setFailure(null)
    setAttempt((current) => current + 1)
  }, [])

  const fresh = Boolean(key) && held.key === key
  return {
    payload: fresh || (Boolean(subject) && held.subject === subject) ? held.payload : null,
    settled: fresh && held.payload !== null,
    error: failure && failure.key === key ? failure.message : null,
    retry,
  }
}
