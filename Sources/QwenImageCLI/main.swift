import Darwin
import Dispatch
import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import QwenImage
import Metal

struct FileLogHandler: LogHandler {
  let label: String
  var metadata: Logger.Metadata = [:]
  var logLevel: Logger.Level
  private let fileHandle: FileHandle?

  init(label: String, fileURL: URL, level: Logger.Level) {
    self.label = label
    self.logLevel = level
    let fm = FileManager.default
    var handle: FileHandle? = nil
    let directory = fileURL.deletingLastPathComponent()
    try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
    if !fm.fileExists(atPath: fileURL.path) {
      fm.createFile(atPath: fileURL.path, contents: nil)
    }
    if let fh = try? FileHandle(forWritingTo: fileURL) {
      fh.seekToEndOfFile()
      handle = fh
    }
    self.fileHandle = handle
  }

  subscript(metadataKey key: String) -> Logger.Metadata.Value? {
    get { metadata[key] }
    set { metadata[key] = newValue }
  }

  func log(
    level: Logger.Level,
    message: Logger.Message,
    metadata: Logger.Metadata?,
    source: String,
    file: String,
    function: String,
    line: UInt
  ) {
    guard let fileHandle else { return }
    var combined = self.metadata
    if let metadata {
      combined.merge(metadata) { _, new in new }
    }
    let metaDescription = combined.isEmpty ? "" : " \(combined)"
    let lineString = "\(Date()) [\(level)] [\(label)] \(message)\(metaDescription)\n"
    if let data = lineString.data(using: .utf8) {
      fileHandle.write(data)
    }
  }
}

LoggingSystem.bootstrap { label in
  let env = ProcessInfo.processInfo.environment
  let levelString = env["QWEN_IMAGE_LOG_LEVEL"]?.lowercased() ?? "info"
  let level = Logger.Level(rawValue: levelString) ?? .info

  var stderrHandler = StreamLogHandler.standardError(label: label)
  stderrHandler.logLevel = level

  if let logPath = env["QWEN_IMAGE_LOG_FILE"] {
    let fileURL = URL(fileURLWithPath: logPath)
    let fileHandler = FileLogHandler(label: label, fileURL: fileURL, level: level)
    return MultiplexLogHandler([stderrHandler, fileHandler])
  } else {
    return stderrHandler
  }
}

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

struct QwenImageCLIEntry {
  static var logger: Logger = {
    var logger = Logger(label: "qwen.image.cli")
    logger.logLevel = .info
    return logger
  }()
  static func run() throws {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX may run on CPU.")
    }
    let cacheLimitBytes = 2 * 1024 * 1024 * 1024
    MLX.GPU.set(cacheLimit: cacheLimitBytes)
    logger.info("Set MLX GPU cache limit to \(ByteCountFormatter.string(fromByteCount: Int64(cacheLimitBytes), countStyle: .memory))")
    Device.setDefault(device: .gpu)
    do {
      let current = Device.defaultDevice()
      logger.info("MLX default device: \(current)")
    }
    var prompt: String?
    var negativePrompt: String?
    var steps = 30
    var guidance: Float = 4.0
    var width = 1024
    var height = 1024
    var editResolution: Int?
    var seed: UInt64?
    var modelArg: String?
    var loraArg: String?
    var revision = "main"
    var outputPath = "qwen-image.png"
    var trueCFGScale: Float?
    var referenceImagePaths: [String] = []
    var quantBits: Int?
    var quantGroupSize = 64
    var quantMode: QuantizationMode = .affine
    var quantAttnBits: Int?
    var quantAttnGroupSize: Int?
    var quantAttnMode: QuantizationMode?
    var quantAttnSpecOverride: QwenQuantizationSpec?
    var quantMinDim = 4096
    var quantizeSnapshotOut: String?
    var quantizeComponents: [String] = ["transformer", "text_encoder"]
    var pythonCmd: String? = nil
    var quantizeScriptPath: String? = nil
    var quantizeSnapshotSwiftOut: String?

    let args = CommandLine.arguments.dropFirst()
    var iterator = args.makeIterator()
    while let argument = iterator.next() {
      switch argument {
      case "--prompt", "-p":
        prompt = nextValue(for: argument, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: argument, iterator: &iterator)
      case "--reference-image":
        referenceImagePaths.append(nextValue(for: argument, iterator: &iterator))
      case "--steps", "-s":
        let value = nextValue(for: argument, iterator: &iterator)
        if let intValue = Int(value) {
          steps = max(1, intValue)
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--guidance", "-g":
        let value = nextValue(for: argument, iterator: &iterator)
        if let floatValue = Float(value) {
          guidance = max(0, floatValue)
        } else {
          fail("Expected float value after \(argument)")
        }
      case "--width", "-W":
        let value = nextValue(for: argument, iterator: &iterator)
        if let intValue = Int(value) {
          width = max(16, intValue)
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--height", "-H":
        let value = nextValue(for: argument, iterator: &iterator)
        if let intValue = Int(value) {
          height = max(16, intValue)
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--edit-resolution":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let intValue = Int(value), intValue == 512 || intValue == 1024 else {
          fail("Expected 512 or 1024 after --edit-resolution")
        }
        editResolution = intValue
      case "--seed":
        let value = nextValue(for: argument, iterator: &iterator)
        if let uintValue = UInt64(value) {
          seed = uintValue
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--model":
        modelArg = nextValue(for: argument, iterator: &iterator)
      case "--lora":
        loraArg = nextValue(for: argument, iterator: &iterator)
      case "--revision":
        revision = nextValue(for: argument, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: argument, iterator: &iterator)
      case "--true-cfg-scale":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let scale = Float(value) else {
          fail("Expected float value after \(argument)")
        }
        trueCFGScale = scale
      case "--quant-bits":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let bits = Int(value), bits == 4 || bits == 8 else {
          fail("Expected 4 or 8 after --quant-bits")
        }
        quantBits = bits
      case "--quant-group-size":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let group = Int(value), group > 0 else {
          fail("Expected positive integer after --quant-group-size")
        }
        quantGroupSize = group
      case "--quant-mode":
        let value = nextValue(for: argument, iterator: &iterator).lowercased()
        switch value {
        case "affine":
          quantMode = .affine
        case "mxfp4":
          quantMode = .mxfp4
        default:
          fail("Unsupported quantization mode \(value). Use affine or mxfp4.")
        }
      case "--quant-min-dim":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let dim = Int(value), dim > 0 else {
          fail("Expected positive integer after --quant-min-dim")
        }
        quantMinDim = dim
      case "--quant-attn":
        quantAttnBits = quantAttnBits ?? quantBits ?? 8
      case "--quant-attn-bits":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let bits = Int(value), bits == 4 || bits == 8 else {
          fail("Expected 4 or 8 after --quant-attn-bits")
        }
        quantAttnBits = bits
      case "--quant-attn-group-size":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let group = Int(value), group > 0 else {
          fail("Expected positive integer after --quant-attn-group-size")
        }
        quantAttnGroupSize = group
      case "--quant-attn-mode":
        let value = nextValue(for: argument, iterator: &iterator).lowercased()
        switch value {
        case "affine":
          quantAttnMode = .affine
        case "mxfp4":
        quantAttnMode = .mxfp4
        default:
          fail("Unsupported quantization mode \(value). Use affine or mxfp4.")
        }
      case "--quantize-snapshot-out":
        quantizeSnapshotOut = nextValue(for: argument, iterator: &iterator)
      case "--quantize-model":
        quantizeSnapshotSwiftOut = nextValue(for: argument, iterator: &iterator)
      case "--quantize-components":
        let value = nextValue(for: argument, iterator: &iterator)
        quantizeComponents = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
      case "--python":
        pythonCmd = nextValue(for: argument, iterator: &iterator)
      case "--quantize-script":
        quantizeScriptPath = nextValue(for: argument, iterator: &iterator)
      case "--help", "-h":
        printUsage()
        return
      default:
        continue
      }
    }

    if quantAttnBits != nil || quantAttnGroupSize != nil || quantAttnMode != nil {
      let bits = quantAttnBits ?? quantBits ?? 8
      let group = quantAttnGroupSize ?? quantGroupSize
      let mode = quantAttnMode ?? quantMode
      quantAttnSpecOverride = QwenQuantizationSpec(groupSize: group, bits: bits, mode: mode)
    }

    if let outDir = quantizeSnapshotSwiftOut {
      guard let modelValue = modelArg else {
        fail("--quantize-model requires --model to point at a local snapshot directory or HF repo id.")
      }
      let env = ProcessInfo.processInfo.environment
      let hfHomePath = env["HF_HOME"].map { NSString(string: $0).expandingTildeInPath }
      let cacheOverridePath: String? = hfHomePath.map { $0 + "/hub" }
      let resolvedSnapshot = try resolveSnapshot(
        model: modelValue,
        revision: revision,
        cacheDirectory: cacheOverridePath,
        hfToken: nil,
        offlineMode: false,
        useBackgroundSession: false
      )
      try runQuantizeSnapshotSwift(
        modelPath: resolvedSnapshot.path,
        outputPath: outDir,
        components: quantizeComponents,
        bits: quantBits ?? 8,
        groupSize: quantGroupSize,
        mode: quantMode
      )
      return
    }

    if let outDir = quantizeSnapshotOut {
      guard let modelValue = modelArg else {
        fail("--quantize-snapshot-out requires --model to point at a local snapshot directory or HF repo id.")
      }
      let env = ProcessInfo.processInfo.environment
      let hfHomePath = env["HF_HOME"].map { NSString(string: $0).expandingTildeInPath }
      let cacheOverridePath: String? = hfHomePath.map { $0 + "/hub" }
      let resolvedSnapshot = try resolveSnapshot(
        model: modelValue,
        revision: revision,
        cacheDirectory: cacheOverridePath,
        hfToken: nil,
        offlineMode: false,
        useBackgroundSession: false
      )
      try runQuantizeSnapshot(
        modelPath: resolvedSnapshot.path,
        outputPath: outDir,
        components: quantizeComponents,
        bits: quantBits ?? 8,
        groupSize: quantGroupSize,
        mode: quantMode,
        pythonCmd: pythonCmd,
        scriptPath: quantizeScriptPath
      )
      return
    }

    guard let prompt else {
      fail("Missing required --prompt")
    }
    logger.debug("Using prompt: \(prompt)")

    let env = ProcessInfo.processInfo.environment
    let hfHomePath = env["HF_HOME"].map { NSString(string: $0).expandingTildeInPath }
    let cacheOverridePath: String? = hfHomePath.map { $0 + "/hub" }

    let snapshotRoot = try resolveSnapshot(
      model: modelArg,
      revision: revision,
      cacheDirectory: cacheOverridePath,
      hfToken: nil,
      offlineMode: false,
      useBackgroundSession: false
    )
    let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
    try prepareOutputDirectory(for: outputURL)

    let generation = GenerationParameters(
      prompt: prompt,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      negativePrompt: negativePrompt,
      seed: seed,
      trueCFGScale: trueCFGScale,
      editResolution: editResolution
    )
    
    let isEdit = !referenceImagePaths.isEmpty
    let pipeline = QwenImagePipeline(config: isEdit ? .imageEditing : .textToImage)
    pipeline.setBaseDirectory(snapshotRoot)
    if let bits = quantBits {
      let spec = QwenQuantizationSpec(groupSize: quantGroupSize, bits: bits, mode: quantMode)
      pipeline.setRuntimeQuantization(spec, minInputDim: quantMinDim)
      logger.info("Enabled runtime quantization (bits: \(bits), group: \(quantGroupSize), mode: \(quantMode), min-dim: \(quantMinDim))")
    }

    let transformerConfig = QwenTransformerConfiguration()
    var modelConfig = QwenModelConfiguration()
    do {
      if let flowMatchConfig = try QwenFlowMatchConfig.load(fromSchedulerDirectory: snapshotRoot.appending(path: "scheduler")) {
        modelConfig.flowMatch = flowMatchConfig
        modelConfig.requiresSigmaShift = flowMatchConfig.useDynamicShifting
        logger.info("Loaded FlowMatch scheduler config (dynamic shifting: \(flowMatchConfig.useDynamicShifting))")
      }
    } catch {
      logger.warning("Failed to load FlowMatch scheduler config: \(error). Falling back to defaults.")
    }

    logger.info("Loading tokenizer and text encoder from \(snapshotRoot.path)")
    try pipeline.prepareTokenizer(from: snapshotRoot, maxLength: nil)
    try pipeline.prepareTextEncoder(from: snapshotRoot)
    if isEdit {
      try pipeline.prepareVAE(from: snapshotRoot)
    }

    if let lora = loraArg {
      let loraURL = try resolveLoraSafetensors(
        lora: lora,
        cacheDirectory: cacheOverridePath,
        hfToken: nil,
        offlineMode: false,
        useBackgroundSession: false
      )
      logger.info("Queuing LoRA for lazy application from \(loraURL.path)")
      pipeline.setPendingLora(from: loraURL)
    }

    if let spec = quantAttnSpecOverride {
      pipeline.setAttentionQuantization(spec)
      logger.info("Enabled attention quantization override (bits: \(spec.bits), group: \(spec.groupSize), mode: \(spec.mode)).")
    }

    let pixels: MLXArray
    if !isEdit {
      logger.info("Generating image")
      pixels = try pipeline.generatePixels(
        parameters: generation,
        model: modelConfig,
        seed: seed
      )
    } else {
#if canImport(CoreGraphics)
        guard !referenceImagePaths.isEmpty else {
          fail("Edit mode requires at least one --reference-image.")
        }
        let primaryReference = loadCGImage(at: referenceImagePaths[0])
        logger.info("Using edit canvas \(generation.width)x\(generation.height)")
        logger.info("Generating image edit")
        if referenceImagePaths.count == 1 {
          pixels = try pipeline.generateEditedPixels(
            parameters: generation,
            model: modelConfig,
            referenceImage: primaryReference,
            maxPromptLength: nil,
            seed: seed
          )
        } else {
          var images: [CGImage] = [primaryReference]
          if referenceImagePaths.count > 2 {
            logger.warning("Received \(referenceImagePaths.count) reference images; only the first two will be used.")
          }
          let second = loadCGImage(at: referenceImagePaths[1])
          images.append(second)
          pixels = try pipeline.generateEditedPixels(
            parameters: generation,
            model: modelConfig,
            referenceImages: images,
            maxPromptLength: nil,
            seed: seed
          )
        }
#else
        fail("Native edit mode requires CoreGraphics support on this platform.")
#endif
    }

    let image = try pipeline.makeImage(from: pixels)
    try save(image: image, to: outputURL)
    logger.info("Saved image to \(outputURL.path)")
  }

  private static func nextValue(
    for option: String,
    iterator: inout IndexingIterator<ArraySlice<String>>
  ) -> String {
    guard let value = iterator.next() else {
      fail("Expected a value after \(option)")
    }
    return value
  }

  private static func resolveSnapshot(
    model: String?,
    revision: String,
    cacheDirectory: String?,
    hfToken: String?,
    offlineMode: Bool,
    useBackgroundSession: Bool
  ) throws -> URL {
    if let model {
      let url = URL(fileURLWithPath: model).standardizedFileURL
      if FileManager.default.fileExists(atPath: url.path) {
        return url
      }
      let cacheURL = cacheDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
      let options = QwenModelRepository.snapshotOptions(
        repoId: model,
        revision: revision,
        cacheDirectory: cacheURL,
        hfToken: hfToken,
        offline: offlineMode,
        useBackgroundSession: useBackgroundSession
      )
      logger.info("Resolving snapshot for \(model) (revision: \(revision))")
      let snapshotURL = try downloadSnapshot(options: options)
      logger.info("Snapshot ready at \(snapshotURL.path)")
      return snapshotURL
    }
    fail("Model not provided. Pass --model with a local path or HF repo id.")
  }

  private static func resolveLoraSafetensors(
    lora: String,
    cacheDirectory: String?,
    hfToken: String?,
    offlineMode: Bool,
    useBackgroundSession: Bool
  ) throws -> URL {
    let fm = FileManager.default
    let localURL = URL(fileURLWithPath: lora).standardizedFileURL

    var isDir: ObjCBool = false
    let exists = fm.fileExists(atPath: localURL.path, isDirectory: &isDir)
    if exists {
      if isDir.boolValue {
        let contents = try fm.contentsOfDirectory(at: localURL, includingPropertiesForKeys: nil)
        if let safetensors = contents.first(where: { $0.pathExtension == "safetensors" }) {
          return safetensors
        }
        fail("No .safetensors file found in LoRA directory \(localURL.path)")
      } else if localURL.pathExtension == "safetensors" {
        return localURL
      } else {
        fail("LoRA path \(localURL.path) is not a .safetensors file")
      }
    }

    let cacheURL = cacheDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
    let options = HubSnapshotOptions(
      repoId: lora,
      revision: "main",
      patterns: ["*.safetensors", "**/*.safetensors"],
      cacheDirectory: cacheURL,
      hfToken: hfToken,
      offline: offlineMode,
      useBackgroundSession: useBackgroundSession
    )
    logger.info("Resolving LoRA snapshot for \(lora)")
    let snapshotRoot = try downloadSnapshot(options: options)

    var snapshotIsDir: ObjCBool = false
    if fm.fileExists(atPath: snapshotRoot.path, isDirectory: &snapshotIsDir) {
      if !snapshotIsDir.boolValue, snapshotRoot.pathExtension == "safetensors" {
        return snapshotRoot
      }
    }

    let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
    if let enumerator = fm.enumerator(
      at: snapshotRoot,
      includingPropertiesForKeys: resourceKeys,
      options: [.skipsHiddenFiles]
    ) {
      for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        return url
      }
    }
    fail("No .safetensors file found in LoRA snapshot at \(snapshotRoot.path)")
  }

  private static func prepareOutputDirectory(for url: URL) throws {
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  }

  private static func downloadSnapshot(options: HubSnapshotOptions) throws -> URL {
    let snapshot = try HubSnapshot(options: options)
    let progressFlag = MutableBox(false)
    let url = try blockingAwait {
      try await snapshot.prepare { progress in
        guard progress.totalUnitCount > 0 else { return }
        progressFlag.value = true
        Self.printDownloadProgress(progress)
      }
    }
    if progressFlag.value {
      let data = Data("\n".utf8)
      try? FileHandle.standardError.write(contentsOf: data)
    }
    return url
  }

  private static func blockingAwait<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let resultBox = MutableBox<Result<T, Error>?>(nil)
    Task {
      do {
        let value = try await operation()
        resultBox.value = .success(value)
      } catch {
        resultBox.value = .failure(error)
      }
      semaphore.signal()
    }
    semaphore.wait()
    guard let result = resultBox.value else {
      fatalError("Snapshot task completed without a result.")
    }
    return try result.get()
  }

  private static func printDownloadProgress(_ progress: HubSnapshotProgress) {
    let percent = progress.totalUnitCount > 0
      ? min(100, max(0, progress.fractionCompleted * 100))
      : 0
    let completed = formatByteCount(progress.completedUnitCount)
    let total = progress.totalUnitCount > 0 ? formatByteCount(progress.totalUnitCount) : "?"
    let speedString: String
    if let speed = progress.estimatedSpeedBytesPerSecond {
      speedString = " @ \(formatSpeed(speed))"
    } else {
      speedString = ""
    }
    let message = String(
      format: "[qwen.image] Downloading snapshot: %@/%@ (%.1f%%)%@",
      completed,
      total,
      percent,
      speedString
    )
    let data = Data("\r\(message)".utf8)
    try? FileHandle.standardError.write(contentsOf: data)
  }

  private static func formatByteCount(_ count: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
  }

  private static func formatSpeed(_ speed: Double) -> String {
    let count = Int64(speed)
    return "\(formatByteCount(count))/s"
  }

  private final class MutableBox<Value>: @unchecked Sendable {
    var value: Value
    init(_ value: Value) {
      self.value = value
    }
  }

  private static func save(image: PipelineImage, to url: URL) throws {
#if canImport(AppKit)
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
      fail("Failed to extract CGImage from NSImage.")
    }
    let ext = url.pathExtension.lowercased()
    let bitmap = NSBitmapImageRep(cgImage: cgImage)
    let fileType: NSBitmapImageRep.FileType
    switch ext {
    case "png": fileType = .png
    case "jpg", "jpeg": fileType = .jpeg
    case "tif", "tiff": fileType = .tiff
    case "bmp": fileType = .bmp
    default: fileType = .png
    }
    let props: [NSBitmapImageRep.PropertyKey: Any] = [:]
    guard let data = bitmap.representation(using: fileType, properties: props) else {
      fail("Failed to encode image as \(fileType)")
    }
    try data.write(to: url)
#elseif canImport(UIKit)
    guard let cgImage = image.cgImage else {
      fail("Failed to extract CGImage from UIImage.")
    }
    let uiImage = UIImage(cgImage: cgImage)
    let ext = url.pathExtension.lowercased()
    let data: Data?
    switch ext {
    case "png": data = uiImage.pngData()
    case "jpg", "jpeg": data = uiImage.jpegData(compressionQuality: 0.95)
    default: data = uiImage.pngData()
    }
    guard let final = data else { fail("Failed to encode image data.") }
    try final.write(to: url)
#else
    fail("Image saving is not supported on this platform.")
#endif
  }

#if canImport(CoreGraphics)
  private static func loadCGImage(at path: String) -> CGImage {
#if canImport(AppKit)
    guard let nsImage = NSImage(contentsOfFile: path) else {
      fail("Failed to load reference image at \(path)")
    }
    var proposedRect = NSRect(origin: .zero, size: nsImage.size)
    guard let cgImage = nsImage.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) else {
      fail("Failed to extract CGImage from reference image at \(path)")
    }
    return cgImage
#elseif canImport(UIKit)
    guard let uiImage = UIImage(contentsOfFile: path), let cgImage = uiImage.cgImage else {
      fail("Failed to load reference image at \(path)")
    }
    return cgImage
#else
    fail("Loading reference images is not supported on this platform.")
#endif
  }
#endif

  private static func printUsage() {
    let usage = """
    qwen-image-cli
      --prompt, -p            Prompt text (required)
      --negative-prompt, --np Negative prompt (optional)
      --steps, -s             Number of diffusion steps (default: 30)
      --guidance, -g          Guidance scale (default: 4.0)
      --true-cfg-scale        True CFG scale (matches diffusers `true_cfg_scale`)
      --width, -W             Image width (default: 1024)
      --height, -H            Image height (default: 1024)
      --seed                  Random seed (optional)
      --model                 HF repo ID (e.g. Qwen/Qwen-Image-Edit-2509) or local snapshot path
      --lora                  Optional LoRA safetensors path or HF repo ID (e.g. Osrivers/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors)
      --revision              Snapshot revision/tag/commit (used with HF repo IDs; default: main)
      --output, -o            Output PNG path (default: qwen-image.png)
      
      Output format is inferred from extension: .png, .bmp, .tiff, .jpg
      Use .bmp or .tiff for faster saves during benchmarking.
      
      Quantize + save snapshot:
      --quantize-model DIR               Write a pre-packed snapshot to DIR (Swift-side; quantizes all Linear layers that pass group-size checks)
      --quantize-components LIST         Comma-separated components to quantize (default: transformer,text_encoder)
      --quant-bits                       Bit-width for snapshot weights (4 or 8; default: 8)
      --quant-group-size                 Group size for snapshot weights (default: 64)
      --quant-mode                       Quantization mode for snapshot weights (affine or mxfp4; default: affine)
      
      Edit mode:
      Pass --reference-image to enable edit mode (repeat to pass two images)
      
      Environment:
      QWEN_IMAGE_LOG_LEVEL   Minimum log level (debug, info, warning, error; default: info)
      QWEN_IMAGE_LOG_FILE    Optional path to a log file; if unset, logs are written to stderr only
      
      --help, -h              Show this help
    """
    if let data = (usage + "\n").data(using: .utf8) {
      try? FileHandle.standardOutput.write(contentsOf: data)
    }
  }

  private static func fail(_ message: String) -> Never {
    logger.error("\(message)")
    exit(2)
  }

  private static func collectLinearLayerRegistry(
    modelPath: String
  ) throws -> [String: [String]] {
    let snapshotURL = URL(fileURLWithPath: modelPath)
    let pipeline = QwenImagePipeline(config: .textToImage)
    let transformerConfig = QwenTransformerConfiguration()
    try pipeline.prepareTextEncoder(from: snapshotURL)
    try pipeline.prepareUNet(from: snapshotURL, configuration: transformerConfig)
    let registry = LinearLayerRegistry.snapshotAndReset()
    return registry
  }

  private static func runQuantizeSnapshot(
    modelPath: String,
    outputPath: String,
    components: [String],
    bits: Int,
    groupSize: Int,
    mode: QuantizationMode,
    pythonCmd: String?,
    scriptPath: String?
  ) throws {
    let snapshotURL = URL(fileURLWithPath: modelPath)
    let outputURL = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

    let scriptURL: URL
    if let scriptPath {
      scriptURL = URL(fileURLWithPath: scriptPath).standardizedFileURL
    } else {
      scriptURL = URL(fileURLWithPath: "quantize_unet_snapshot.py").standardizedFileURL
    }
    let scriptPathResolved = scriptURL.path
    let py = pythonCmd ?? "python3"
    var args: [String] = []
    args.append(contentsOf: [py, scriptPathResolved, "--snapshot", snapshotURL.path, "--output", outputURL.path])
    args.append(contentsOf: ["--group-size", String(groupSize), "--bits", String(bits)])
    switch mode {
    case .mxfp4: args.append(contentsOf: ["--mode", "mxfp4"]) 
    default: args.append(contentsOf: ["--mode", "affine"]) 
    }
    for comp in components { args.append(contentsOf: ["--component", comp]) }
    args.append("--no-dry-run")

    let fullCmd: [String]
    if pythonCmd?.contains(" ") ?? false {
      let parts = pythonCmd!.split(separator: " ").map(String.init)
      let modeArg = mode == .mxfp4 ? "mxfp4" : "affine"
      let componentArgs = components.flatMap { ["--component", $0] }
      fullCmd = parts
        + [
          scriptPathResolved,
          "--snapshot", snapshotURL.path,
          "--output", outputURL.path,
          "--group-size", String(groupSize),
          "--bits", String(bits),
          "--mode", modeArg
        ]
        + componentArgs
        + ["--no-dry-run"]
    } else {
      fullCmd = args
    }

    logger.info("Quantizing snapshot (bits=\(bits), group=\(groupSize), mode=\(mode), components=\(components.joined(separator: ",")))")
    let (status, stdout, stderr) = runProcess(command: fullCmd)
    if status != 0 {
      logger.error("Quantizer failed (exit=\(status))\nSTDOUT:\n\(stdout)\nSTDERR:\n\(stderr)")
      fail("Quantizer failed with status \(status)")
    }
    logger.info("Quantized snapshot written to \(outputURL.path). Use --model \(outputURL.path) in subsequent runs.")
  }

  @discardableResult
  private static func runProcess(command: [String]) -> (Int32, String, String) {
    let task = Process()
    task.standardOutput = Pipe()
    task.standardError = Pipe()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    task.arguments = command
    do { try task.run() } catch {
      return (127, "", String(describing: error))
    }
    let outData = (task.standardOutput as! Pipe).fileHandleForReading.readDataToEndOfFile()
    let errData = (task.standardError as! Pipe).fileHandleForReading.readDataToEndOfFile()
    task.waitUntilExit()
    let stdout = String(data: outData, encoding: .utf8) ?? ""
    let stderr = String(data: errData, encoding: .utf8) ?? ""
    return (task.terminationStatus, stdout, stderr)
  }

  private static func runQuantizeSnapshotSwift(
    modelPath: String,
    outputPath: String,
    components: [String],
    bits: Int,
    groupSize: Int,
    mode: QuantizationMode
  ) throws {
    let snapshotURL = URL(fileURLWithPath: modelPath)
    let outputURL = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
    let spec = QwenQuantizationSpec(groupSize: groupSize, bits: bits, mode: mode)
    logger.info("Swift quantize+save (bits=\(bits), group=\(groupSize), mode=\(mode), components=\(components.joined(separator: ",")))")
    var allowedMap: [String: Set<String>] = [:]
    let registry = try collectLinearLayerRegistry(modelPath: modelPath)
    for (comp, list) in registry {
      allowedMap[comp] = Set(list)
    }
    func canonicalize(_ name: String, component: String) -> String {
      var n = name
      if component == "text_encoder" {
        if n.hasPrefix("encoder.") { n = "model." + n.dropFirst("encoder.".count) }
      }
      n = n.replacingOccurrences(of: ".img_ff.mlp_in", with: ".img_mlp.net.0.proj")
      n = n.replacingOccurrences(of: ".img_ff.mlp_out", with: ".img_mlp.net.2")
      n = n.replacingOccurrences(of: ".txt_ff.mlp_in", with: ".txt_mlp.net.0.proj")
      n = n.replacingOccurrences(of: ".txt_ff.mlp_out", with: ".txt_mlp.net.2")
      n = n.replacingOccurrences(of: ".attn.attn_to_out", with: ".attn.to_out.0")
      return n
    }
    var allowedMapHF: [String: Set<String>] = [:]
    for (comp, set) in allowedMap {
      var out = Set<String>()
      for name in set { out.insert(canonicalize(name, component: comp)) }
      allowedMapHF[comp] = out
    }
    try SwiftQuantSaver.quantizeAndSave(from: snapshotURL, to: outputURL, components: components, spec: spec, allowedLayerMap: allowedMapHF)
    let rsyncCmd = [
      "rsync", "-aL",
      "--exclude", "transformer",
      "--exclude", "text_encoder",
      "--exclude", "quantization.json",
      snapshotURL.path + "/",
      outputURL.path + "/"
    ]
    let (rc, out, err) = runProcess(command: rsyncCmd)
    if rc != 0 {
      logger.warning("Ancillary copy via rsync failed (exit=\(rc)): \(err)\n\(out)")
    }
    logger.info("Swift quantized snapshot written to \(outputURL.path)")
  }
}

do {
  try QwenImageCLIEntry.run()
} catch {
  QwenImageCLIEntry.logger.error("Unhandled error: \(error)")
  exit(1)
}
