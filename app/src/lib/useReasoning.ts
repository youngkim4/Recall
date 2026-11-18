import { useCallback, useMemo, useSyncExternalStore } from 'react'
import { choiceOf } from './models'
import { effortOf, readEfforts, reasoningSteps, storeEffort, subscribeEfforts, type ReasoningStep } from './reasoning'

type Reasoning = {
  // the levels the model takes, least first; none for a model with no levels
  steps: ReasoningStep[]
  // the level a question to the model asks for; '' for none
  effort: string
  setEffort: (level: string) => void
}

const nothingKept = () => readEfforts()

// The reasoning level of one model: what it can be set to, what it is set
// to, and the way to set it. Every control and every question reads the same
// kept level, so they agree.
export function useReasoning(model: string): Reasoning {
  const kept = useSyncExternalStore(subscribeEfforts, readEfforts, nothingKept)
  const choice = choiceOf(model)
  const steps = useMemo(() => reasoningSteps(choice), [choice])
  const setEffort = useCallback((level: string) => storeEffort(model, level), [model])
  return { steps, effort: effortOf(choice, kept), setEffort }
}
