const MODEL_LABELS: Record<string, string> = {
  'gpt-5.5': 'Best quality',
  'gpt-5.4': 'Balanced',
  'gpt-5.4-mini': 'Lower cost',
}

const MODEL_DESCRIPTIONS: Record<string, string> = {
  'gpt-5.5': 'Best long-context quality for full reports.',
  'gpt-5.4': 'Balanced quality, speed, and cost.',
  'gpt-5.4-mini': 'Cheaper quick pass for smaller threads.',
}

export function modelLabel(model: string) {
  return MODEL_LABELS[model] ? `${MODEL_LABELS[model]} (${model})` : model
}

export function modelDescription(model: string) {
  return MODEL_DESCRIPTIONS[model] || 'Custom model from server settings.'
}
