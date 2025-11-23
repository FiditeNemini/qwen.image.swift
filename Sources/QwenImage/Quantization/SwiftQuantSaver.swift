import Foundation
import MLX
import MLXNN

public enum SwiftQuantSaverError: Error {
  case missingComponentDirectory(String, URL)
  case noSafetensorsFound(URL)
}

public struct SwiftQuantSaver {
  private static let allowedSuffixesTransformer: [String] = [
    ".img_mlp.net.0.proj",
    ".img_mlp.net.2",
    ".txt_mlp.net.0.proj",
    ".txt_mlp.net.2",
    "img_in",
    "txt_in",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "norm_out.linear",
    "proj_out"
  ]

  private static let allowedSuffixesTextEncoder: [String] = [
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj"
  ]

  private static func isAllowed(base: String, component: String) -> Bool {
    let suffixes = (component == "text_encoder") ? allowedSuffixesTextEncoder : allowedSuffixesTransformer
    for s in suffixes where base.hasSuffix(s) { return true }
    return false
  }

  public static func quantizeAndSave(
    from sourceRoot: URL,
    to outputRoot: URL,
    components: [String],
    spec: QwenQuantizationSpec,
    allowedLayerMap: [String: Set<String>]? = nil
  ) throws {
    try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)

    var layers: [(component: String, name: String, shape: [Int], inDim: Int, outDim: Int, file: String)] = []

    for component in components {
      switch component {
      case "transformer":
        let src = sourceRoot.appending(path: "transformer")
        let dst = outputRoot.appending(path: "transformer")
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let fileName = try quantizeAndSaveComponent(
          componentName: "transformer",
          inputDirectory: src,
          outputDirectory: dst,
          baseFileName: "diffusion_pytorch_model.safetensors",
          spec: spec,
          allowedLayerMap: allowedLayerMap,
          record: &layers
        )
        _ = fileName
      case "text_encoder":
        let src = sourceRoot.appending(path: "text_encoder")
        let dst = outputRoot.appending(path: "text_encoder")
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let fileName = try quantizeAndSaveComponent(
          componentName: "text_encoder",
          inputDirectory: src,
          outputDirectory: dst,
          baseFileName: "model.safetensors",
          spec: spec,
          allowedLayerMap: allowedLayerMap,
          record: &layers
        )
        _ = fileName
      default:
        continue
      }
    }

    try writeManifest(
      at: outputRoot,
      sourceRoot: sourceRoot,
      spec: spec,
      layers: layers
    )

    try copyAncillary(from: sourceRoot, to: outputRoot)
  }

  private static func collectSafetensors(in directory: URL) throws -> [URL] {
    let fm = FileManager.default
    guard fm.fileExists(atPath: directory.path) else {
      throw SwiftQuantSaverError.missingComponentDirectory(directory.lastPathComponent, directory)
    }
    let items = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    let files = items.filter { $0.pathExtension == "safetensors" }
    guard !files.isEmpty else { throw SwiftQuantSaverError.noSafetensorsFound(directory) }
    return files.sorted { $0.path < $1.path }
  }

  @discardableResult
  private static func quantizeAndSaveComponent(
    componentName: String,
    inputDirectory: URL,
    outputDirectory: URL,
    baseFileName: String,
    spec: QwenQuantizationSpec,
    allowedLayerMap: [String: Set<String>]?,
    record layers: inout [(component: String, name: String, shape: [Int], inDim: Int, outDim: Int, file: String)]
  ) throws -> String {
    let shardURLs = try collectSafetensors(in: inputDirectory)
    let readers = try shardURLs.map { try SafeTensorsReader(fileURL: $0) }
    var merged = try WeightsMapping.merge(readers: readers, dtype: nil)

    let supportedGroupSizes: Set<Int> = [32, 64, 128]
    let group = spec.groupSize
    guard supportedGroupSizes.contains(group) else { return baseFileName }

    var quantizedBases = Set<String>()
    let strict = (allowedLayerMap != nil)
    let allowedSet = allowedLayerMap?[componentName]
    for (key, tensor) in merged {
      guard key.hasSuffix(".weight") else { continue }
      guard tensor.ndim == 2 else { continue }
      let outDim = tensor.dim(0)
      let inDim = tensor.dim(1)
      let base = String(key.dropLast(".weight".count))
      if strict {
        guard let allowed = allowedSet, allowed.contains(base) else { continue }
      } else {
        guard isAllowed(base: base, component: componentName) else { continue }
      }
      guard inDim % group == 0 else { continue }
      var f = tensor
      if f.dtype != .float32 && f.dtype != .bfloat16 && f.dtype != .float16 { continue }
      f = f.asType(.float32)
      let (wq, scales, biases) = MLX.quantized(f, groupSize: group, bits: spec.bits, mode: spec.mode)
      merged[key] = wq
      merged["\(base).scales"] = scales
      if let b = biases { merged["\(base).biases"] = b }
      quantizedBases.insert(base)
      layers.append((componentName, base, [outDim, inDim], inDim, outDim, ""))
    }

    func bytes(of array: MLXArray) -> Int {
      let count = array.shape.reduce(1, *)
      return count * array.dtype.size
    }
    let maxShardBytes = 4_500_000_000 // < 5 GB including metadata
    let ordered = merged.keys.sorted().map { ($0, merged[$0]!) }
    var chunks: [[(String, MLXArray)]] = []
    var current: [(String, MLXArray)] = []
    var currentBytes = 0
    for (k, v) in ordered {
      let sz = bytes(of: v)
      if currentBytes > 0 && currentBytes + sz > maxShardBytes {
        chunks.append(current)
        current = []
        currentBytes = 0
      }
      current.append((k, v))
      currentBytes += sz
    }
    if !current.isEmpty { chunks.append(current) }

    let total = max(1, chunks.count)
    let stem = String(baseFileName.split(separator: ".").first ?? Substring(baseFileName))
    for (i, chunk) in chunks.enumerated() {
      let shardName = String(format: "%@-%05d-of-%05d.safetensors", stem, i + 1, total)
      let dst = outputDirectory.appending(path: shardName)
      var dict: [String: MLXArray] = [:]
      for (k, v) in chunk { dict[k] = v }
      try MLX.save(arrays: dict, metadata: [:], url: dst)
    }

    var fileMap: [String: String] = [:]
    for (i, chunk) in chunks.enumerated() {
      let shardName = String(format: "%@-%05d-of-%05d.safetensors", stem, i + 1, total)
      let keys = Set(chunk.map { $0.0 })
      for base in quantizedBases where keys.contains("\(base).weight") { fileMap[base] = shardName }
    }
    for idx in 0..<layers.count {
      if layers[idx].component == componentName, let fname = fileMap[layers[idx].name] {
        layers[idx].file = fname
      }
    }
    return baseFileName
  }

  private static func writeManifest(
    at root: URL,
    sourceRoot: URL,
    spec: QwenQuantizationSpec,
    layers: [(component: String, name: String, shape: [Int], inDim: Int, outDim: Int, file: String)]
  ) throws {
    var payload: [String: Any] = [:]
    payload["version"] = 1
    payload["snapshot"] = sourceRoot.path
    payload["group_size"] = spec.groupSize
    payload["bits"] = spec.bits
    payload["mode"] = (spec.mode == .mxfp4 ? "mxfp4" : "affine")

    var layerArray: [[String: Any]] = []
    layerArray.reserveCapacity(layers.count)
    for entry in layers {
      var rec: [String: Any] = [:]
      rec["component"] = entry.component
      rec["name"] = entry.name
      rec["file"] = entry.file
      rec["shape"] = entry.shape
      rec["in_dim"] = entry.inDim
      rec["out_dim"] = entry.outDim
      rec["quant_file"] = entry.file
      rec["group_size"] = spec.groupSize
      rec["bits"] = spec.bits
      rec["mode"] = (spec.mode == .mxfp4 ? "mxfp4" : "affine")
      layerArray.append(rec)
    }
    payload["layers"] = layerArray

    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    let url = root.appending(path: "quantization.json")
    try data.write(to: url)
  }

  private static func copyAncillary(from src: URL, to dst: URL) throws {
    let fm = FileManager.default
    let entries = try fm.contentsOfDirectory(atPath: src.path)
    let skipDirs: Set<String> = ["transformer", "text_encoder"]
    let copyDirs: Set<String> = ["tokenizer", "processor", "scheduler", "vae", "text_encoder_2"]
    let copyFileExts: Set<String> = ["json", "md", "txt"]
    for name in entries {
      if skipDirs.contains(name) { continue }
      var srcPath = src.appending(path: name)
      let dstPath = dst.appending(path: name)
      var isDir: ObjCBool = false
      if fm.fileExists(atPath: srcPath.path, isDirectory: &isDir) {
        if let rv = try? srcPath.resourceValues(forKeys: [.isSymbolicLinkKey]), rv.isSymbolicLink == true {
          srcPath = srcPath.resolvingSymlinksInPath()
          _ = fm.fileExists(atPath: srcPath.path, isDirectory: &isDir)
        }
        if isDir.boolValue {
          guard copyDirs.contains(name) else { continue }
          try? fm.removeItem(at: dstPath)
          try fm.copyItem(at: srcPath, to: dstPath)
        } else {
          let ext = srcPath.pathExtension.lowercased()
          guard copyFileExts.contains(ext) else { continue }
          if name == "quantization.json" { continue }
          try? fm.removeItem(at: dstPath)
          try fm.copyItem(at: srcPath, to: dstPath)
        }
      }
    }
  }
}
