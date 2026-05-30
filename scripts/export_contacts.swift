import Contacts
import Foundation

let outputPath = CommandLine.arguments.dropFirst().first ?? "saves/contact_names.json"
let store = CNContactStore()
let semaphore = DispatchSemaphore(value: 0)
var granted = false
var accessError: Error?

store.requestAccess(for: .contacts) { ok, error in
    granted = ok
    accessError = error
    semaphore.signal()
}

_ = semaphore.wait(timeout: .now() + 60)

if !granted {
    let message = accessError?.localizedDescription ?? "Contacts access was not granted"
    fputs("Contacts access denied: \(message)\n", stderr)
    exit(2)
}

let keys: [CNKeyDescriptor] = [
    CNContactGivenNameKey as CNKeyDescriptor,
    CNContactMiddleNameKey as CNKeyDescriptor,
    CNContactFamilyNameKey as CNKeyDescriptor,
    CNContactNicknameKey as CNKeyDescriptor,
    CNContactOrganizationNameKey as CNKeyDescriptor,
    CNContactPhoneNumbersKey as CNKeyDescriptor,
    CNContactEmailAddressesKey as CNKeyDescriptor,
]

let request = CNContactFetchRequest(keysToFetch: keys)
var contacts: [[String: Any]] = []

try store.enumerateContacts(with: request) { contact, _ in
    let name = [
        contact.givenName,
        contact.middleName,
        contact.familyName,
    ]
    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    .filter { !$0.isEmpty }
    .joined(separator: " ")

    let fallbackName = [
        contact.nickname,
        contact.organizationName,
    ]
    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    .first { !$0.isEmpty } ?? ""

    let displayName = name.isEmpty ? fallbackName : name
    if displayName.isEmpty {
        return
    }

    let phones = contact.phoneNumbers.map { $0.value.stringValue }
    let emails = contact.emailAddresses.map { String($0.value) }
    if phones.isEmpty && emails.isEmpty {
        return
    }

    contacts.append([
        "name": displayName,
        "phones": phones,
        "emails": emails,
    ])
}

let payload: [String: Any] = [
    "exportedAt": ISO8601DateFormatter().string(from: Date()),
    "contacts": contacts,
]
let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
let outputURL = URL(fileURLWithPath: outputPath)
try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
try data.write(to: outputURL)
print("Exported \(contacts.count) contacts to \(outputPath)")
