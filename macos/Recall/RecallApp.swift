import Cocoa
import Foundation
import UniformTypeIdentifiers
import WebKit

@main
final class RecallApp: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKScriptMessageHandler {
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

        // the wizard re-probes the instant the user returns from System Settings
        NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.webView?.evaluateJavaScript(
                "window.dispatchEvent(new CustomEvent('recall:becameActive'))"
            )
        }

        do {
            let bundledServer = bundledServerExecutable()
            serverPort = Self.portForThisLaunch(
                rememberedAs: bundledServer == nil ? Self.developmentPortKey : Self.packagedPortKey
            )
            if let bundledServer {
                // packaged app: self-contained server, data in App Support
                let dataDir = try ensureDataDirectory()
                try startBundledBackend(executable: bundledServer, dataDir: dataDir, port: serverPort)
            } else {
                let appRoot = try resolveAppRoot()
                let python = try resolvePython(appRoot: appRoot)
                try startBackend(appRoot: appRoot, python: python, port: serverPort)
            }
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

    // MARK: - Web bridge

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard message.name == "recall",
              let body = message.body as? [String: Any],
              let cmd = body["cmd"] as? String else {
            return
        }
        let requestId = body["id"] as? String ?? ""
        switch cmd {
        case "openFullDiskAccess":
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles") {
                NSWorkspace.shared.open(url)
            }
            resolveBridge(requestId, json: "true")
        case "pickDatabase":
            pickDatabaseFile(requestId: requestId)
        case "relaunch":
            resolveBridge(requestId, json: "true")
            relaunchApp()
        case "print":
            resolveBridge(requestId, json: "true")
            saveAsPDF()
        case "tick":
            // a control with steps went from one step to the next: one tick of the trackpad
            NSHapticFeedbackManager.defaultPerformer.perform(.levelChange, performanceTime: .now)
        default:
            resolveBridge(requestId, json: "null")
        }
    }

    private func resolveBridge(_ requestId: String, json: String) {
        guard !requestId.isEmpty, let webView else { return }
        let script = "window.__recallBridge && window.__recallBridge._resolve('\(requestId)', \(json))"
        DispatchQueue.main.async {
            webView.evaluateJavaScript(script)
        }
    }

    private func pickDatabaseFile(requestId: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let panel = NSOpenPanel()
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowsMultipleSelection = false
            // ~/Library is hidden in the panel by default; a chat.db copy may sit there
            panel.showsHiddenFiles = true
            panel.treatsFilePackagesAsDirectories = true
            panel.message = "Choose a copy of your Messages database (chat.db)"
            panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser
            panel.begin { response in
                if response == .OK, let url = panel.url {
                    let escaped = url.path
                        .replacingOccurrences(of: "\\", with: "\\\\")
                        .replacingOccurrences(of: "'", with: "\\'")
                    self.resolveBridge(requestId, json: "'\(escaped)'")
                } else {
                    self.resolveBridge(requestId, json: "null")
                }
            }
        }
    }

    // MARK: - PDF

    /// The page asked to print itself. Recall has one thing to print, a report
    /// set as a document, and here that means a file: the place to save it is
    /// asked for, and the page goes through the print system into a PDF there,
    /// in pages of the user's paper size, laid out by the page's print styles.
    private func saveAsPDF() {
        DispatchQueue.main.async { [weak self] in
            guard let self, let webView = self.webView, let window = self.window else { return }
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.pdf]
            panel.canCreateDirectories = true
            panel.nameFieldStringValue = self.fileName(from: webView.title) + ".pdf"
            panel.beginSheetModal(for: window) { response in
                guard response == .OK, let url = panel.url,
                      let info = NSPrintInfo.shared.copy() as? NSPrintInfo else {
                    self.printingEnded()
                    return
                }
                info.jobDisposition = .save
                info.dictionary()[NSPrintInfo.AttributeKey.jobSavingURL] = url
                // The page sets its own margins (@page). Half an inch here is
                // what puts the first line of a page where those margins say.
                info.topMargin = 36
                info.bottomMargin = 36
                info.leftMargin = 36
                info.rightMargin = 36
                let operation = webView.printOperation(with: info)
                operation.showsPrintPanel = false
                operation.showsProgressPanel = true
                // pages are laid out in the operation's view, which has no size until it is given one
                operation.view?.frame = NSRect(origin: .zero, size: info.paperSize)
                operation.runModal(
                    for: window,
                    delegate: self,
                    didRun: #selector(self.printOperationDidRun(_:success:contextInfo:)),
                    contextInfo: nil
                )
            }
        }
    }

    @objc private func printOperationDidRun(
        _ operation: NSPrintOperation,
        success: Bool,
        contextInfo: UnsafeMutableRawPointer?
    ) {
        printingEnded()
    }

    /// A page that began a print hears that it is over, saved or not.
    private func printingEnded() {
        webView?.evaluateJavaScript("window.dispatchEvent(new Event('afterprint'))")
    }

    /// The page's title as a file name: nothing in it that a path cannot hold.
    private func fileName(from title: String?) -> String {
        let cleaned = (title ?? "")
            .components(separatedBy: CharacterSet(charactersIn: "/:\\\n"))
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)
        return cleaned.isEmpty ? "Recall report" : cleaned
    }

    private func relaunchApp() {
        // FDA applies at process launch; a clean child teardown first means
        // no orphan keeps serving under the old TCC identity
        stopBackend()
        let bundlePath = Bundle.main.bundleURL.path.replacingOccurrences(of: "'", with: "'\\''")
        let helper = Process()
        helper.executableURL = URL(fileURLWithPath: "/bin/sh")
        helper.arguments = ["-c", "sleep 0.4; /usr/bin/open '\(bundlePath)'"]
        try? helper.run()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.terminate(nil)
        }
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        // x-apple.systempreferences and friends must open natively, never navigate the app
        if let url = navigationAction.request.url,
           let scheme = url.scheme?.lowercased(),
           scheme != "http", scheme != "https", scheme != "about" {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
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
        // The persistent store keeps localStorage (saved chats, saved messages,
        // model/path/layout prefs) between launches, one store for each origin.
        // The origin holds the port, so the page finds its state again only
        // when it opens on the same port: see portForThisLaunch.
        configuration.websiteDataStore = WKWebsiteDataStore.default()
        // native bridge for the onboarding wizard: FDA deep-link, file picker, relaunch
        configuration.userContentController.add(self, name: "recall")
        // A web view does nothing when its page calls window.print(). Here the
        // call comes to the shell, which saves the page as a PDF.
        configuration.userContentController.addUserScript(WKUserScript(
            source: """
            window.__RECALL_SHELL__ = true;
            window.print = function () {
              window.webkit.messageHandlers.recall.postMessage({ cmd: 'print', id: '' });
            };
            """,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        ))

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

    private func bundledServerExecutable() -> URL? {
        let url = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers/RecallServer.app/Contents/MacOS/RecallServer")
        return FileManager.default.isExecutableFile(atPath: url.path) ? url : nil
    }

    private func bundledExporterApp() -> URL? {
        var candidates: [URL] = [
            Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/Recall Contacts Exporter.app")
        ]
        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appendingPathComponent("Recall Contacts Exporter.app"))
        }
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private func ensureDataDirectory() throws -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dataDir = appSupport.appendingPathComponent("Recall", isDirectory: true)
        try FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)
        return dataDir
    }

    private func startBundledBackend(executable: URL, dataDir: URL, port: Int) throws {
        let process = Process()
        // direct Process launch (not `open`) keeps TCC attributing the child's
        // chat.db reads to Recall.app, so the user's FDA grant covers it
        process.executableURL = executable
        process.currentDirectoryURL = dataDir

        var environment = ProcessInfo.processInfo.environment
        environment["RECALL_UI_HOST"] = "127.0.0.1"
        environment["RECALL_UI_PORT"] = String(port)
        environment["RECALL_DATA_DIR"] = dataDir.path
        environment["RECALL_BUNDLED"] = "1"
        if let exporter = bundledExporterApp() {
            environment["RECALL_CONTACTS_EXPORTER_APP"] = exporter.path
        }
        process.environment = environment

        try process.run()
        serverProcess = process
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
        if let exporter = bundledExporterApp() {
            environment["RECALL_CONTACTS_EXPORTER_APP"] = exporter.path
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
                } else if self.startupAttempts < 240 {
                    // first launch of the packaged app pays Gatekeeper's full
                    // bundle scan plus a cold pandas import -- budget ~60s
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
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

    // MARK: - Server port

    /// Where a port is remembered in the app's defaults. The packaged app and
    /// the development build read two archives, so each has a port, and with
    /// it a store, of its own.
    static let packagedPortKey = "RecallServerPort"
    static let developmentPortKey = "RecallServerPortDev"
    /// The dynamic ports. Each Mac draws its own: no port is the same for everybody.
    static let serverPortRange = 49152...65535
    /// How many ports are drawn before the last resort.
    static let portDraws = 100
    /// The server's own default (ui_server.py), for when no drawn port is free.
    static let lastResortPort = 8765
    /// A remembered port that is in use gets 30 checks with 0.1 s between them: about 3 s.
    static let rememberedPortChecks = 30
    static let rememberedPortPause: TimeInterval = 0.1

    /// The port of this launch: the remembered port when it is free, so that
    /// the page opens on the same origin and finds its saved state again (see
    /// createWindow).
    ///
    /// The first port that is drawn is remembered in `defaults` under `key`,
    /// and it stays. When it is in use at a later launch (a second copy of the
    /// app, or a server of the last launch that has not stopped), that launch
    /// gets a drawn port, where the page does not find its saved state, and
    /// the next launch asks for the remembered port again. To remember the
    /// drawn port in its place would leave all saved state behind for good.
    static func portForThisLaunch(
        rememberedAs key: String,
        in defaults: UserDefaults = .standard,
        isFree: (Int) -> Bool = RecallApp.isPortAvailable,
        pause: () -> Void = { Thread.sleep(forTimeInterval: RecallApp.rememberedPortPause) }
    ) -> Int {
        let remembered = defaults.integer(forKey: key)
        let chosen = choosePort(
            remembered: remembered,
            isFree: isFree,
            draw: { Int.random(in: serverPortRange) },
            pause: pause
        )
        guard let port = chosen else {
            // the last resort is not remembered: the next launch chooses again
            return lastResortPort
        }
        if !serverPortRange.contains(remembered) {
            defaults.set(port, forKey: key)
        }
        return port
    }

    /// Chooses the port of the server. `remembered` is the port in the
    /// defaults; a number outside the range means that there is none.
    ///
    /// The remembered port is taken when it is free. When it is in use, it is
    /// asked about again after each `pause`, because the server of the last
    /// launch needs time to let go of it: about 0.5 s after SIGTERM when no
    /// job runs, and in the packaged app at most 2 s after the shell is gone,
    /// also when the shell crashed or a job still runs (`main` in
    /// ui_server.py). The server of the development build has no such limit.
    /// After the last check a port is drawn.
    ///
    /// The result is nil when no drawn port is free.
    static func choosePort(
        remembered: Int,
        isFree: (Int) -> Bool,
        draw: () -> Int,
        pause: () -> Void
    ) -> Int? {
        if serverPortRange.contains(remembered) {
            for check in 0..<rememberedPortChecks {
                if check > 0 {
                    pause()
                }
                if isFree(remembered) {
                    return remembered
                }
            }
        }
        for _ in 0..<portDraws {
            let port = draw()
            if isFree(port) {
                return port
            }
        }
        return nil
    }

    // nonisolated: it only binds a socket, and portForThisLaunch takes it as a plain function
    private nonisolated static func isPortAvailable(_ port: Int) -> Bool {
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
