// The server offers a few models. The interface offers them as quality
// levels, lowest first, with the model's own name beside each.
const LEVELS = [
  { model: 'gpt-5.4-mini', label: 'Low', note: 'Fastest, lowest cost' },
  { model: 'gpt-5.4', label: 'Medium', note: 'A balance of speed and care' },
  { model: 'gpt-5.5', label: 'High', note: 'Most careful, highest cost' },
]

export type QualityOption = {
  // the model's id, which is what is sent to the server
  value: string
  label: string
  // the model as its maker writes it: "GPT-5.4 mini"
  name: string
  // what choosing it means, in a few words; '' for a model with no level
  note: string
}

// "gpt-5.4-mini" -> "GPT-5.4 mini". Only a numbered GPT model is written
// this way; any other, such as "gpt-oss-120b", keeps the name it came with.
function modelName(model: string) {
  if (!/^gpt-\d/i.test(model)) return model
  return model.replace(/^gpt-/i, 'GPT-').replace(/-(?=[a-z])/gi, ' ')
}

// the levels this server can run, lowest first; a model with no level keeps its own name
export function qualityOptions(models: string[]): QualityOption[] {
  const levelled = LEVELS.filter((level) => models.includes(level.model)).map((level) => ({
    value: level.model,
    label: level.label,
    name: modelName(level.model),
    note: level.note,
  }))
  const other = models
    .filter((model) => !LEVELS.some((level) => level.model === model))
    .map((model) => ({ value: model, label: modelName(model), name: '', note: '' }))
  return [...levelled, ...other]
}
