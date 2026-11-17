// The rules of the setup guide, with no screen in them.

// What the guide asks the server for when it starts an import. The archive
// file is the one that the app reads; with none set, the server writes its
// own. A database is named only when a copy was chosen in this guide.
export function importRequest(input: { archivePath: string; copyPath: string }) {
  return {
    messagesPath: input.archivePath.trim() || undefined,
    dbPath: input.copyPath || undefined,
    includeContacts: true,
  }
}
