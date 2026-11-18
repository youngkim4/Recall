// one choice in a short menu (ChoiceMenu)
export type Choice = {
  value: string
  label: string
  // a quiet word beside the label, such as the name of the model behind a level
  meta?: string
  // what choosing it means, in a few words
  note?: string
}
