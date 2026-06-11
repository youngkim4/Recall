import Cocoa
import Foundation
import WebKit

@main
final class RecallApp: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private static var sharedDelegate: RecallApp?

    private var window: NSWindow?
    private var webView: WKWebView?
    private var serverProcess: Process?
    private var serverPort: Int = 0
    private var startupAttempts = 0

    static func main() {
        let app = NSApplication.shared
        let delegate = RecallApp()
        sharedDelegate = delegate
        app.delegate = delegate
        app.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        setApplicationIcon()
        buildMenu()

        do {
            let appRoot = try resolveAppRoot()
            let python = try resolvePython(appRoot: appRoot)
            serverPort = findAvailablePort()
            try startBackend(appRoot: appRoot, python: python, port: serverPort)
            createWindow()
            waitForBackend()
        } catch {
            createWindow()
            showErrorPage(error.localizedDescription)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopBackend()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func buildMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenu.addItem(NSMenuItem(
            title: "Hide Recall",
            action: #selector(NSApplication.hide(_:)),
            keyEquivalent: "h"
        ))
        let hideOthers = NSMenuItem(
            title: "Hide Others",
            action: #selector(NSApplication.hideOtherApplications(_:)),
            keyEquivalent: "h"
        )
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(hideOthers)
        appMenu.addItem(.separator())
        appMenu.addItem(NSMenuItem(
            title: "Quit Recall",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        ))
        appMenuItem.submenu = appMenu

        // without an Edit menu, cmd+A/C/V/X/Z never reach the web view
        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(NSMenuItem(title: "Undo", action: Selector(("undo:")), keyEquivalent: "z"))
        let redo = NSMenuItem(title: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(redo)
        editMenu.addItem(.separator())
        editMenu.addItem(NSMenuItem(title: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x"))
        editMenu.addItem(NSMenuItem(title: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c"))
        editMenu.addItem(NSMenuItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v"))
        editMenu.addItem(NSMenuItem(
            title: "Select All",
            action: #selector(NSText.selectAll(_:)),
            keyEquivalent: "a"
        ))
        editMenuItem.submenu = editMenu

        let windowMenuItem = NSMenuItem()
        mainMenu.addItem(windowMenuItem)
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(NSMenuItem(
            title: "Minimize",
            action: #selector(NSWindow.performMiniaturize(_:)),
            keyEquivalent: "m"
        ))
        windowMenu.addItem(NSMenuItem(title: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: ""))
        windowMenu.addItem(.separator())
        windowMenu.addItem(NSMenuItem(
            title: "Close",
            action: #selector(NSWindow.performClose(_:)),
            keyEquivalent: "w"
        ))
        windowMenuItem.submenu = windowMenu
        NSApp.windowsMenu = windowMenu

        NSApp.mainMenu = mainMenu
    }

    private func setApplicationIcon() {
        guard let iconURL = Bundle.main.url(forResource: "RecallIcon", withExtension: "icns"),
              let icon = NSImage(contentsOf: iconURL) else {
            return
        }
        NSApp.applicationIconImage = icon
    }

    private func createWindow() {
        if window != nil {
            return
        }

        let configuration = WKWebViewConfiguration()
        // Persistent store so localStorage (saved chats, model/path/layout prefs) survives app restarts.
        configuration.websiteDataStore = WKWebsiteDataStore.default()

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        self.webView = webView

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1440, height: 920),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Recall"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.contentView = webView
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = window
    }

    private func resolveAppRoot() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["RECALL_APP_ROOT"], !value.isEmpty {
            return URL(fileURLWithPath: value).standardizedFileURL
        }

        if let rootPath = Bundle.main.path(forResource: "RecallRoot", ofType: "path") {
            let value = try String(contentsOfFile: rootPath, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty {
                return URL(fileURLWithPath: value).standardizedFileURL
            }
        }

        if let resourceURL = Bundle.main.resourceURL {
            let bundledCore = resourceURL.appendingPathComponent("RecallCore")
            if FileManager.default.fileExists(atPath: bundledCore.appendingPathComponent("ui_server.py").path) {
                return bundledCore.standardizedFileURL
            }
        }

        throw RecallAppError.missingAppRoot
    }

    private func resolvePython(appRoot: URL) throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        if let value = environment["RECALL_PYTHON"], !value.isEmpty {
            let python = URL(fileURLWithPath: value).standardizedFileURL
            if FileManager.default.isExecutableFile(atPath: python.path) {
                return python
            }
        }

        let venvPython = appRoot.appendingPathComponent("venv/bin/python")
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return venvPython
        }

        let systemPython = URL(fileURLWithPath: "/usr/bin/python3")
        if FileManager.default.isExecutableFile(atPath: systemPython.path) {
            return systemPython
        }

        throw RecallAppError.missingPython
    }

    private func startBackend(appRoot: URL, python: URL, port: Int) throws {
        let serverScript = appRoot.appendingPathComponent("ui_server.py")
        guard FileManager.default.fileExists(atPath: serverScript.path) else {
            throw RecallAppError.missingServer(serverScript.path)
        }

        let process = Process()
        process.executableURL = python
        process.arguments = [serverScript.path]
        process.currentDirectoryURL = appRoot

        var environment = ProcessInfo.processInfo.environment
        environment["RECALL_UI_HOST"] = "127.0.0.1"
        environment["RECALL_UI_PORT"] = String(port)
        environment["PYTHONUNBUFFERED"] = "1"
        if let resourceURL = Bundle.main.resourceURL {
            let exporter = resourceURL.appendingPathComponent("Recall Contacts Exporter.app")
            if FileManager.default.fileExists(atPath: exporter.path) {
                environment["RECALL_CONTACTS_EXPORTER_APP"] = exporter.path
            }
        }
        process.environment = environment

        let logURL = logFileURL()
        try FileManager.default.createDirectory(
            at: logURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        if let logHandle = try? FileHandle(forWritingTo: logURL) {
            logHandle.seekToEndOfFile()
            process.standardOutput = logHandle
            process.standardError = logHandle
        }

        try process.run()
        serverProcess = process
    }

    private func waitForBackend() {
        startupAttempts += 1
        let url = URL(string: "http://127.0.0.1:\(serverPort)/api/defaults")!
        URLSession.shared.dataTask(with: url) { [weak self] _, response, _ in
            guard let self else { return }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            DispatchQueue.main.async {
                if status == 200 {
                    self.loadApp()
                } else if self.startupAttempts < 80 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                        self.waitForBackend()
                    }
                } else {
                    self.showErrorPage("Recall could not start its local backend. Check ~/Library/Logs/Recall/RecallBackend.log.")
                }
            }
        }.resume()
    }

    private func loadApp() {
        guard let webView else { return }
        let version = Int(Date().timeIntervalSince1970)
        let url = URL(string: "http://127.0.0.1:\(serverPort)/?app=mac&v=\(version)")!
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30))
    }

    private func showErrorPage(_ message: String) {
        let escaped = htmlEscape(message)
        let html = """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { margin: 0; font: 16px -apple-system, BlinkMacSystemFont, "Inter", sans-serif; background: #f5f5f5; color: #27272a; }
            main { max-width: 720px; margin: 120px auto; background: white; border: 1px solid #e4e4e7; border-radius: 12px; padding: 32px; }
            h1 { margin: 0 0 12px; font-size: 24px; }
            p { color: #71717a; line-height: 1.5; }
            code { background: #f4f4f5; border: 1px solid #e4e4e7; border-radius: 6px; padding: 2px 6px; }
          </style>
        </head>
        <body>
          <main>
            <h1>Recall could not open</h1>
            <p>\(escaped)</p>
            <p>Backend logs are written to <code>~/Library/Logs/Recall/RecallBackend.log</code>.</p>
          </main>
        </body>
        </html>
        """
        webView?.loadHTMLString(html, baseURL: nil)
    }

    private func stopBackend() {
        guard let process = serverProcess else { return }
        if process.isRunning {
            process.terminate()
            DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) {
                if process.isRunning {
                    process.interrupt()
                }
            }
        }
        serverProcess = nil
    }

    private func findAvailablePort() -> Int {
        for _ in 0..<100 {
            let port = Int.random(in: 49152...65535)
            if isPortAvailable(port) {
                return port
            }
        }
        return 8765
    }

    private func isPortAvailable(_ port: Int) -> Bool {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        if descriptor < 0 {
            return false
        }
        defer { close(descriptor) }

        var reuse: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")

        return withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }

    private func logFileURL() -> URL {
        let library = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
        return library.appendingPathComponent("Logs/Recall/RecallBackend.log")
    }

    private func htmlEscape(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
    }
}

enum RecallAppError: LocalizedError {
    case missingAppRoot
    case missingPython
    case missingServer(String)

    var errorDescription: String? {
        switch self {
        case .missingAppRoot:
            "Could not find the Recall app root. Rebuild the app from the project folder."
        case .missingPython:
            "Could not find Python. Run the setup command from README.md, then rebuild the app."
        case .missingServer(let path):
            "Could not find ui_server.py at \(path). Rebuild the app from the project folder."
        }
    }
}
