// The server offers a few models and says how each one ranks. The interface
// offers them as quality levels, lowest first, with the model's own name
// beside each. No model is known here by its id.
import type { ModelChoice } from '../types'

// what a level is called, and what choosing it means in a few words
const LEVELS: Record<string, { label: string; note: string }> = {
  low: { label: 'Low', note: 'Fastest, lowest cost' },
  medium: { label: 'Medium', note: 'A balance of speed and care' },
  high: { label: 'High', note: 'Most careful, highest cost' },
  top: { label: 'Top', note: '' },
}

export type QualityOption = {
  // the model's id, which is what is sent to the server
  value: string
  label: string
  // the model as its maker writes it; '' for a model with no level
  name: string
  // what choosing it means, in a few words; '' where there is nothing to say
  note: string
}

// The choices the server last gave. A control that has only the ids of the
// models reads the names and levels from here.
let offered: ModelChoice[] = []

// Keeps what /api/defaults said of its models. Call it before the defaults
// are put on the page, so the first menu already has its levels.
export function offerModels(choices: ModelChoice[] | undefined) {
  offered = Array.isArray(choices) ? [...choices] : []
}

// What the server last said of one model, for a control that has only its id.
export function choiceOf(model: string, choices: ModelChoice[] = offered): ModelChoice | undefined {
  return choices.find((choice) => choice.id === model)
}

function levelled(choice: ModelChoice): QualityOption {
  const level = LEVELS[choice.level]
  // a level this build does not know: the model's own name says the most
  if (!level) return { value: choice.id, label: choice.label || choice.id, name: '', note: '' }
  return { value: choice.id, label: level.label, name: choice.label, note: level.note }
}

// The levels among `models`, in the order the server gives them, which is
// lowest first. A model the server does not describe keeps its id.
export function qualityOptions(models: string[], choices: ModelChoice[] = offered): QualityOption[] {
  const known = Array.isArray(choices) ? choices.filter((choice) => models.includes(choice.id)) : []
  const other = models
    .filter((model) => !known.some((choice) => choice.id === model))
    .map((model) => ({ value: model, label: model, name: '', note: '' }))
  return [...known.map(levelled), ...other]
}
