import Foundation
import MLX
import MLXNN

public enum QwenWeightsLoaderError: Error {
  case missingTextEncoderDirectory(URL)
  case missingTransformerDirectory(URL)
  case missingVAEDirectory(URL)
  case missingVisionDirectory(URL)
  case noSafetensorsFound(URL)
}

public struct QwenWeightsLoader {
  public init() {}

  public func loadTextEncoder(
    from snapshot: HubSnapshot,
    into textEncoder: QwenTextEncoder,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil,
    progress: HubSnapshot.ProgressHandler? = nil
  ) async throws {
    let root = try await snapshot.prepare(progressHandler: progress)
    try loadTextEncoder(
      fromDirectory: root,
      into: textEncoder,
      dtype: dtype,
      quantization: quantization
    )
  }

  public func loadTextEncoder(
    fromDirectory root: URL,
    into textEncoder: QwenTextEncoder,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let directory = root.appending(path: "text_encoder")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingTextEncoderDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)
    applyQuantization(
      plan: quantization,
      to: textEncoder,
      availableKeys: availableKeys,
      tensorNameTransform: Self.textEncoderTensorName
    )
    let parameters = try WeightsMapping.textEncoderParameters(
      from: merged,
      dtype: dtype,
      quantization: quantization
    )
    textEncoder.update(parameters: parameters)
  }

  private static func collectSafetensors(in directory: URL) throws -> [URL] {
    let fileManager = FileManager.default
    let contents = try fileManager.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    )
    return contents.filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
  }

  public func loadTransformer(
    from snapshot: HubSnapshot,
    into transformer: QwenTransformer,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil,
    progress: HubSnapshot.ProgressHandler? = nil
  ) async throws {
    let root = try await snapshot.prepare(progressHandler: progress)
    try loadTransformer(
      fromDirectory: root,
      into: transformer,
      dtype: dtype,
      quantization: quantization
    )
  }

  public func loadTransformer(
    fromDirectory root: URL,
    into transformer: QwenTransformer,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let directory = root.appending(path: "transformer")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingTransformerDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)
    applyQuantization(
      plan: quantization,
      to: transformer,
      availableKeys: availableKeys,
      tensorNameTransform: Self.transformerTensorName
    )
    let parameters = try WeightsMapping.transformerParameters(
      from: merged,
      configuration: transformer.configuration,
      dtype: dtype,
      quantization: quantization
    )
    transformer.update(parameters: parameters)
  }

  public func loadUNet(
    from snapshot: HubSnapshot,
    into unet: QwenUNet,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil,
    progress: HubSnapshot.ProgressHandler? = nil
  ) async throws {
    try await loadTransformer(
      from: snapshot,
      into: unet.transformer,
      dtype: dtype,
      quantization: quantization,
      progress: progress
    )
  }

  public func loadUNet(
    fromDirectory root: URL,
    into unet: QwenUNet,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let directory = root.appending(path: "transformer")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingTransformerDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)
    applyQuantization(
      plan: quantization,
      to: unet.transformer,
      availableKeys: availableKeys,
      tensorNameTransform: Self.transformerTensorName
    )
    let parameters = try WeightsMapping.transformerParameters(
      from: merged,
      configuration: unet.transformer.configuration,
      dtype: dtype,
      quantization: quantization
    )
    unet.transformer.update(parameters: parameters)
  }

  public func loadVAE(
    from snapshot: HubSnapshot,
    into vae: QwenVAE,
    dtype: DType? = nil,
    progress: HubSnapshot.ProgressHandler? = nil
  ) async throws {
    let root = try await snapshot.prepare(progressHandler: progress)
    try loadVAE(fromDirectory: root, into: vae, dtype: dtype)
  }

  public func loadVAE(
    fromDirectory root: URL,
    into vae: QwenVAE,
    dtype: DType? = nil
  ) throws {
    let directory = root.appending(path: "vae")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingVAEDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let parameters = try WeightsMapping.vaeParameters(from: merged, dtype: dtype)
    vae.update(parameters: parameters)
  }

  func loadVisionPatchEmbed(
    from snapshot: HubSnapshot,
    into patchEmbed: QwenVisionPatchEmbed,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil,
    progress: HubSnapshot.ProgressHandler? = nil
  ) async throws {
    let root = try await snapshot.prepare(progressHandler: progress)
    try loadVisionPatchEmbed(
      fromDirectory: root,
      into: patchEmbed,
      dtype: dtype,
      quantization: quantization
    )
  }

  func loadVisionPatchEmbed(
    fromDirectory root: URL,
    into patchEmbed: QwenVisionPatchEmbed,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let directory = root.appending(path: "text_encoder")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingVisionDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)
    applyQuantization(
      plan: quantization,
      to: patchEmbed,
      availableKeys: availableKeys,
      tensorNameTransform: Self.visionTensorName
    )
    if let parameters = WeightsMapping.visionPatchEmbedParameters(
      from: merged,
      dtype: dtype,
      quantization
    ) {
      patchEmbed.update(parameters: parameters)
    } else {
      throw WeightsMappingError.missingTensor("visual.patch_embed.proj.weight")
    }
  }

  func loadVisionPatchMerger(
    from snapshot: HubSnapshot,
    into merger: QwenVisionPatchMerger,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil,
    progress: HubSnapshot.ProgressHandler? = nil
  ) async throws {
    let root = try await snapshot.prepare(progressHandler: progress)
    try loadVisionPatchMerger(
      fromDirectory: root,
      into: merger,
      dtype: dtype,
      quantization: quantization
    )
  }

  func loadVisionPatchMerger(
    fromDirectory root: URL,
    into merger: QwenVisionPatchMerger,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let directory = root.appending(path: "text_encoder")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingVisionDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)
    applyQuantization(
      plan: quantization,
      to: merger,
      availableKeys: availableKeys,
      tensorNameTransform: Self.visionTensorName
    )
    if let parameters = WeightsMapping.visionPatchMergerParameters(
      from: merged,
      dtype: dtype,
      quantization: quantization
    ) {
      merger.update(parameters: parameters)
    } else {
      throw WeightsMappingError.missingTensor("visual.merger")
    }
  }

  func loadVisionTower(
    fromDirectory root: URL,
    into tower: QwenVisionTower,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let directory = root.appending(path: "text_encoder")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingVisionDirectory(directory)
    }

    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(directory)
    }

    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)
    applyQuantization(
      plan: quantization,
      to: tower,
      availableKeys: availableKeys,
      tensorNameTransform: Self.visionTensorName
    )

    guard let patchParameters = WeightsMapping.visionPatchEmbedParameters(
      from: merged,
      dtype: dtype,
      quantization
    ) else {
      throw WeightsMappingError.missingTensor("visual.patch_embed.proj.weight")
    }
    tower.updatePatchEmbed(parameters: patchParameters)

    if let mergerParameters = WeightsMapping.visionPatchMergerParameters(
      from: merged,
      dtype: dtype,
      quantization: quantization
    ) {
      tower.updatePatchMerger(parameters: mergerParameters)
    } else {
      throw WeightsMappingError.missingTensor("visual.merger")
    }

    for index in 0..<tower.blockCount {
      if let parameters = WeightsMapping.visionBlockParameters(
        from: merged,
        blockIndex: index,
        dtype: dtype,
        quantization: quantization
      ) {
        tower.updateBlock(at: index, parameters: parameters)
      }
    }
  }

  func visionBlockParameters(
    fromDirectory root: URL,
    blockIndex: Int,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws -> ModuleParameters? {
    let directory = root.appending(path: "text_encoder")
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: directory.path) else {
      throw QwenWeightsLoaderError.missingVisionDirectory(directory)
    }
    let safetensors = try QwenWeightsLoader.collectSafetensors(in: directory)
    guard !safetensors.isEmpty else { return nil }
    let readers = try safetensors.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    return WeightsMapping.visionBlockParameters(
      from: merged,
      blockIndex: blockIndex,
      dtype: dtype,
      quantization: quantization
    )
  }

  private func applyQuantization(
    plan: QwenQuantizationPlan?,
    to model: Module,
    availableKeys: Set<String>,
    tensorNameTransform: (String) -> String
  ) {
    guard let plan, plan.isEnabled else { return }
    quantize(model: model) { path, _ in
      let tensorName = tensorNameTransform(path)
      guard let spec = plan.quantization(for: tensorName) else {
        return nil
      }
      let scalesKey = "\(tensorName).scales"
      guard availableKeys.contains(scalesKey) else {
        return nil
      }
      return spec.tuple
    }
  }

  private static func textEncoderTensorName(_ path: String) -> String {
    if path.hasPrefix("encoder.") {
      let suffix = path.dropFirst("encoder.".count)
      return "model." + suffix
    }
    return path
  }

  private static func visionTensorName(_ path: String) -> String {
    if path.hasPrefix("patch_embed.") {
      let suffix = String(path.dropFirst("patch_embed.".count))
      return "visual.patch_embed." + suffix
    }
    if path.hasPrefix("patch_merger.") {
      let suffix = path.dropFirst("patch_merger.".count)
      switch suffix {
      case "mlp_0":
        return "visual.merger.mlp.0"
      case "mlp_2":
        return "visual.merger.mlp.2"
      default:
        return "visual.merger." + String(suffix)
      }
    }
    if path.hasPrefix("blocks.") {
      var suffix = String(path.dropFirst("blocks.".count))
      suffix = suffix.replacingOccurrences(of: ".mlp.gate", with: ".mlp.gate_proj")
      suffix = suffix.replacingOccurrences(of: ".mlp.up", with: ".mlp.up_proj")
      suffix = suffix.replacingOccurrences(of: ".mlp.down", with: ".mlp.down_proj")
      return "visual.blocks." + suffix
    }
    return path
  }

  private static func transformerTensorName(_ path: String) -> String {
    var name = path
    // Map internal module paths to HF safetensor base keys used in manifests.
    // Feed-forward MLPs
    name = name.replacingOccurrences(of: ".img_ff.mlp_in", with: ".img_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".img_ff.mlp_out", with: ".img_mlp.net.2")
    name = name.replacingOccurrences(of: ".txt_ff.mlp_in", with: ".txt_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".txt_ff.mlp_out", with: ".txt_mlp.net.2")
    // Attention output projection
    name = name.replacingOccurrences(of: ".attn.attn_to_out", with: ".attn.to_out.0")
    // Modulation linears from AdaLayerNorm
    name = name.replacingOccurrences(of: ".img_norm1.mod_linear", with: ".img_mod.1")
    name = name.replacingOccurrences(of: ".txt_norm1.mod_linear", with: ".txt_mod.1")
    return name
  }
}
