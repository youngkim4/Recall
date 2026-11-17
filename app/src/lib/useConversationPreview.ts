import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'
import type { PreviewPayload } from '../types'
import { recallApi } from './api'
import {
  isPreviewPayload,
  makePreviewCacheKey,
  previewGeneration,
  readPreviewCache,
  watchPreviewCache,
  writePreviewCache,
} from './cache'
import { NOTHING_HELD, shownPreview, type Held } from './previewHeld'

type PreviewRequest = {
  // '' for any of the first three means there is nothing to ask for yet
  messagesPath: string
  contact: string
  model: string
  since: string
  until: string
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

// Loads the figures for one conversation, from the browser's store first and
// then from the server. onStored is called when a fresh answer has been kept.
// When the saved figures are cleared (an import does it), the figures held
// here go too and are asked for again: they were of the archive before.
export function useConversationPreview(
  { messagesPath, contact, model, since, until }: PreviewRequest,
  onStored: () => void,
): ConversationPreview {
  const [held, setHeld] = useState<Held>(NOTHING_HELD)
  const [failure, setFailure] = useState<Failure | null>(null)
  const [attempt, setAttempt] = useState(0)
  const generation = useSyncExternalStore(watchPreviewCache, previewGeneration)
  const ready = Boolean(messagesPath && contact && model)
  const key = ready ? makePreviewCacheKey({ messagesPath, contact, model, since, until }) : ''
  const subject = ready ? `${messagesPath}\n${contact}` : ''

  useEffect(() => {
    if (!key) return
    let cancelled = false
    const stored = readPreviewCache(key)
    if (stored) {
      queueMicrotask(() => {
        if (!cancelled) setHeld({ key, subject, generation, payload: stored })
      })
    }
    recallApi
      .preview({ messagesPath, contact, model, since, until })
      .then((payload) => {
        if (!isPreviewPayload(payload)) throw new Error('This conversation could not be read.')
        // asked before the saved figures were cleared: an answer of the archive before
        if (previewGeneration() !== generation) return
        writePreviewCache(key, payload)
        if (cancelled) return
        onStored()
        setHeld({ key, subject, generation, payload })
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
  }, [key, subject, generation, messagesPath, contact, model, since, until, onStored, attempt])

  const retry = useCallback(() => {
    setFailure(null)
    setAttempt((current) => current + 1)
  }, [])

  return {
    ...shownPreview(held, { key, subject, generation }),
    error: failure && failure.key === key ? failure.message : null,
    retry,
  }
}
