// The two lists of the Overview reach the foot of the window: each shows as
// many rows as its box holds (useRowsThatFit), and never fewer than five.
// A row is a band of 44px under a hairline of 1px (lists.css).

export const LIST_FEWEST = 5
// the most rows a list can be asked for: Lately asks the server for this many people
export const LIST_MOST = 20

// the height of one row in pixels, for the size of a rem; one function for the life of the app
export const listRowHeight = (rem: number) => 2.75 * rem + 1
