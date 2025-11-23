import Foundation
import MLX
import MLXNN

public enum LinearLayerRegistry {
  private static var storage: [String: Set<String>] = [:]
  private static var componentStack: [String] = []

  public static func withComponent<T>(
    _ component: String,
    _ block: () throws -> T
  ) rethrows -> T {
    componentStack.append(component)
    defer { componentStack.removeLast() }
    return try block()
  }

  static func record(_ tensorBase: String) {
    guard let component = componentStack.last else { return }
    var set = storage[component] ?? Set<String>()
    set.insert(tensorBase)
    storage[component] = set
  }

  public static func snapshotAndReset() -> [String: [String]] {
    let result = storage.mapValues { Array($0).sorted() }
    storage.removeAll()
    return result
  }
}

public enum WeightsMappingError: Error {
  case duplicateTensor(String)
  case missingTensor(String)
}

public struct WeightsMapping {
  public static func merge(
    readers: [SafeTensorsReader],
    dtype: DType? = nil
  ) throws -> [String: MLXArray] {
    var merged: [String: MLXArray] = [:]
    for reader in readers {
      let tensors = try reader.loadAllTensors(as: dtype)
      for (name, tensor) in tensors {
        if merged[name] != nil {
          throw WeightsMappingError.duplicateTensor(name)
        }
        merged[name] = tensor
      }
    }
    return merged
  }

  public static func textEncoderParameters(
    from tensors: [String: MLXArray],
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws -> ModuleParameters {
    try LinearLayerRegistry.withComponent("text_encoder") {
      var source = filteredTensors(tensors)
      var flat: [String: MLXArray] = [:]
      try assignEmbedding(
        flat: &flat,
        source: &source,
        modulePath: "encoder.embed_tokens",
        tensorBase: "model.embed_tokens",
        dtype: dtype,
        quantization: quantization
      )
      flat["encoder.norm.weight"] = try fetch(
        "model.norm.weight",
        from: &source,
        dtype: dtype
      )

      let layerIndices = collectLayerIndices(from: source.keys)

      for layerIndex in layerIndices {
        let prefix = "model.layers.\(layerIndex)"
        let mlxPrefix = "encoder.layers.\(layerIndex)"

        flat["\(mlxPrefix).input_layernorm.weight"] = try fetch(
          "\(prefix).input_layernorm.weight",
          from: &source,
          dtype: dtype
        )
        flat["\(mlxPrefix).post_attention_layernorm.weight"] = try fetch(
          "\(prefix).post_attention_layernorm.weight",
          from: &source,
          dtype: dtype
        )

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).self_attn.q_proj",
          tensorBase: "\(prefix).self_attn.q_proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).self_attn.k_proj",
          tensorBase: "\(prefix).self_attn.k_proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).self_attn.v_proj",
          tensorBase: "\(prefix).self_attn.v_proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).self_attn.o_proj",
          tensorBase: "\(prefix).self_attn.o_proj",
          hasBias: false,
          dtype: dtype,
          quantization: quantization
        )

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).mlp.gate_proj",
          tensorBase: "\(prefix).mlp.gate_proj",
          hasBias: false,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).mlp.up_proj",
          tensorBase: "\(prefix).mlp.up_proj",
          hasBias: false,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(mlxPrefix).mlp.down_proj",
          tensorBase: "\(prefix).mlp.down_proj",
          hasBias: false,
          dtype: dtype,
          quantization: quantization
        )
      }

      return ModuleParameters.unflattened(flat)
    }
  }

  public static func transformerParameters(
    from tensors: [String: MLXArray],
    configuration: QwenTransformerConfiguration,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) throws -> ModuleParameters {
    try LinearLayerRegistry.withComponent("transformer") {
      var source = tensors
      var flat: [String: MLXArray] = [:]

      try assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "img_in",
        tensorBase: "img_in",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
      flat["txt_norm.weight"] = try fetch("txt_norm.weight", from: &source, dtype: dtype)
      try assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "txt_in",
        tensorBase: "txt_in",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )

      try assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "time_text_embed.timestep_embedder.linear_1",
        tensorBase: "time_text_embed.timestep_embedder.linear_1",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
      try assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "time_text_embed.timestep_embedder.linear_2",
        tensorBase: "time_text_embed.timestep_embedder.linear_2",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )

      try assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "norm_out.linear",
        tensorBase: "norm_out.linear",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
      try assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "proj_out",
        tensorBase: "proj_out",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )

      for blockIndex in 0..<configuration.numberOfLayers {
        let prefix = "transformer_blocks.\(blockIndex)"
        let modulePrefix = "transformer_blocks.\(blockIndex)"

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).img_norm1.mod_linear",
          tensorBase: "\(prefix).img_mod.1",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).txt_norm1.mod_linear",
          tensorBase: "\(prefix).txt_mod.1",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.to_q",
          tensorBase: "\(prefix).attn.to_q",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.to_k",
          tensorBase: "\(prefix).attn.to_k",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.to_v",
          tensorBase: "\(prefix).attn.to_v",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.add_q_proj",
          tensorBase: "\(prefix).attn.add_q_proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.add_k_proj",
          tensorBase: "\(prefix).attn.add_k_proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.add_v_proj",
          tensorBase: "\(prefix).attn.add_v_proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )

        flat["\(modulePrefix).attn.norm_q.weight"] = try fetch(
          "\(prefix).attn.norm_q.weight",
          from: &source,
          dtype: dtype
        )
        flat["\(modulePrefix).attn.norm_k.weight"] = try fetch(
          "\(prefix).attn.norm_k.weight",
          from: &source,
          dtype: dtype
        )
        flat["\(modulePrefix).attn.norm_added_q.weight"] = try fetch(
          "\(prefix).attn.norm_added_q.weight",
          from: &source,
          dtype: dtype
        )
        flat["\(modulePrefix).attn.norm_added_k.weight"] = try fetch(
          "\(prefix).attn.norm_added_k.weight",
          from: &source,
          dtype: dtype
        )

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.attn_to_out",
          tensorBase: "\(prefix).attn.to_out.0",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).attn.to_add_out",
          tensorBase: "\(prefix).attn.to_add_out",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).img_ff.mlp_in",
          tensorBase: "\(prefix).img_mlp.net.0.proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).img_ff.mlp_out",
          tensorBase: "\(prefix).img_mlp.net.2",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )

        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).txt_ff.mlp_in",
          tensorBase: "\(prefix).txt_mlp.net.0.proj",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
        try assignLinear(
          flat: &flat,
          source: &source,
          modulePath: "\(modulePrefix).txt_ff.mlp_out",
          tensorBase: "\(prefix).txt_mlp.net.2",
          hasBias: true,
          dtype: dtype,
          quantization: quantization
        )
      }

      return ModuleParameters.unflattened(flat)
    }
  }

  public static func visionPatchEmbedParameters(
    from tensors: [String: MLXArray],
    dtype: DType? = nil,
    _ quantization: QwenQuantizationPlan? = nil
  ) -> ModuleParameters? {
    if let tensor = tensors["vision_model.patch_embed.proj.weight"] ?? tensors["patch_embed.proj.weight"] ?? tensors["visual.patch_embed.proj.weight"] {
      let converted = convertConv3D(tensor)
      let adjusted = dtypeAdjusted(converted, dtype: dtype)
      let flat: [String: MLXArray] = ["proj.weight": adjusted]
      return ModuleParameters.unflattened(flat)
    }
    return nil
  }

  public static func visionPatchMergerParameters(
    from tensors: [String: MLXArray],
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) -> ModuleParameters? {
    var source = tensors
    guard let normWeight = source.removeValue(forKey: "visual.merger.ln_q.weight") else {
      return nil
    }

    var flat: [String: MLXArray] = [:]
    flat["ln_q.weight"] = dtypeAdjusted(normWeight, dtype: dtype)
    if let normBias = source.removeValue(forKey: "visual.merger.ln_q.bias") {
      flat["ln_q.bias"] = dtypeAdjusted(normBias, dtype: dtype)
    }

    if source["visual.merger.mlp.0.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "mlp_0",
        tensorBase: "visual.merger.mlp.0",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }
    if source["visual.merger.mlp.2.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "mlp_2",
        tensorBase: "visual.merger.mlp.2",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }

    guard !flat.isEmpty else { return nil }
    return ModuleParameters.unflattened(flat)
  }

  public static func visionBlockParameters(
    from tensors: [String: MLXArray],
    blockIndex: Int,
    dtype: DType? = nil,
    quantization: QwenQuantizationPlan? = nil
  ) -> ModuleParameters? {
    let prefix = "visual.blocks.\(blockIndex)"
    guard tensors["\(prefix).attn.qkv.weight"] != nil else {
      return nil
    }

    var source = tensors
    var flat: [String: MLXArray] = [:]

    if let weight = source.removeValue(forKey: "\(prefix).norm1.weight") {
      flat["norm1.weight"] = dtypeAdjusted(weight, dtype: dtype)
    }
    if let weight = source.removeValue(forKey: "\(prefix).norm2.weight") {
      flat["norm2.weight"] = dtypeAdjusted(weight, dtype: dtype)
    }
    if source["\(prefix).attn.qkv.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "attn.qkv",
        tensorBase: "\(prefix).attn.qkv",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }
    if source["\(prefix).attn.proj.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "attn.proj",
        tensorBase: "\(prefix).attn.proj",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }

    if source["\(prefix).mlp.gate_proj.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "mlp.gate",
        tensorBase: "\(prefix).mlp.gate_proj",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }
    if source["\(prefix).mlp.up_proj.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "mlp.up",
        tensorBase: "\(prefix).mlp.up_proj",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }
    if source["\(prefix).mlp.down_proj.weight"] != nil {
      try? assignLinear(
        flat: &flat,
        source: &source,
        modulePath: "mlp.down",
        tensorBase: "\(prefix).mlp.down_proj",
        hasBias: true,
        dtype: dtype,
        quantization: quantization
      )
    }

    guard !flat.isEmpty else { return nil }
    return ModuleParameters.unflattened(flat)
  }

  public static func vaeParameters(
    from tensors: [String: MLXArray],
    dtype: DType? = nil
  ) throws -> ModuleParameters {
    var source = tensors
    var flat: [String: MLXArray] = [:]

    func reshapeNorm(_ tensor: MLXArray, images: Bool) -> MLXArray {
      var result = tensor
      if result.ndim == 1 {
        if images {
          result = result.reshaped(result.dim(0), 1, 1)
        } else {
          result = result.reshaped(result.dim(0), 1, 1, 1)
        }
      }
      return result
    }

    func fetchNormTensor(
      _ prefix: String,
      from tensors: inout [String: MLXArray],
      dtype: DType?,
      images: Bool
    ) throws -> MLXArray {
      if let tensor = try? fetch("\(prefix).weight", from: &tensors, dtype: dtype) {
        return reshapeNorm(tensor, images: images)
      }
      if let tensor = try? fetch("\(prefix).gamma", from: &tensors, dtype: dtype) {
        return reshapeNorm(tensor, images: images)
      }
      throw WeightsMappingError.missingTensor("\(prefix).weight")
    }

    // Decoder ----------------------------------------------------------------

    flat["decoder.conv_in.conv.weight"] = try convertConv3D(
      fetch("decoder.conv_in.weight", from: &source, dtype: dtype)
    )
    flat["decoder.conv_in.conv.bias"] = try fetch("decoder.conv_in.bias", from: &source, dtype: dtype)

    let midResnetIndices = collectNestedIndices(in: source.keys, components: ["decoder", "mid_block", "resnets"])
    for resIndex in midResnetIndices.sorted() {
      let prefix = "decoder.mid_block.resnets.\(resIndex)"
      flat["decoder.mid_block.resnets.\(resIndex).norm1.weight"] = try fetchNormTensor(
        "\(prefix).norm1", from: &source, dtype: dtype, images: false
      )
      flat["decoder.mid_block.resnets.\(resIndex).norm2.weight"] = try fetchNormTensor(
        "\(prefix).norm2", from: &source, dtype: dtype, images: false
      )
      flat["decoder.mid_block.resnets.\(resIndex).conv1.conv.weight"] = try convertConv3D(
        fetch("\(prefix).conv1.weight", from: &source, dtype: dtype)
      )
      flat["decoder.mid_block.resnets.\(resIndex).conv1.conv.bias"] = try fetch(
        "\(prefix).conv1.bias", from: &source, dtype: dtype)
      flat["decoder.mid_block.resnets.\(resIndex).conv2.conv.weight"] = try convertConv3D(
        fetch("\(prefix).conv2.weight", from: &source, dtype: dtype)
      )
      flat["decoder.mid_block.resnets.\(resIndex).conv2.conv.bias"] = try fetch(
        "\(prefix).conv2.bias", from: &source, dtype: dtype)
    }

    let attnIndices = collectNestedIndices(in: source.keys, components: ["decoder", "mid_block", "attentions"])
    for attnIndex in attnIndices.sorted() {
      let prefix = "decoder.mid_block.attentions.\(attnIndex)"
      flat["decoder.mid_block.attentions.\(attnIndex).norm.weight"] = try fetchNormTensor(
        "\(prefix).norm", from: &source, dtype: dtype, images: true
      )
      flat["decoder.mid_block.attentions.\(attnIndex).to_qkv.weight"] = try convertConv2D(
        fetch("\(prefix).to_qkv.weight", from: &source, dtype: dtype)
      )
      flat["decoder.mid_block.attentions.\(attnIndex).to_qkv.bias"] = try fetch(
        "\(prefix).to_qkv.bias", from: &source, dtype: dtype)
      flat["decoder.mid_block.attentions.\(attnIndex).proj.weight"] = try convertConv2D(
        fetch("\(prefix).proj.weight", from: &source, dtype: dtype)
      )
      flat["decoder.mid_block.attentions.\(attnIndex).proj.bias"] = try fetch(
        "\(prefix).proj.bias", from: &source, dtype: dtype)
    }

    let upBlockIndices = collectNestedIndices(in: source.keys, components: ["decoder", "up_blocks"])
    for blockIndex in upBlockIndices.sorted() {
      let resnetPrefix = "decoder.up_blocks.\(blockIndex).resnets"
      let resnetIndices = collectNestedIndices(in: source.keys, components: ["decoder", "up_blocks", "\(blockIndex)", "resnets"])
      for resnetIndex in resnetIndices.sorted() {
        let prefix = "\(resnetPrefix).\(resnetIndex)"
        flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).norm1.weight"] = try fetchNormTensor(
          "\(prefix).norm1", from: &source, dtype: dtype, images: false
        )
        flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).norm2.weight"] = try fetchNormTensor(
          "\(prefix).norm2", from: &source, dtype: dtype, images: false
        )
        flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).conv1.conv.weight"] = try convertConv3D(
          fetch("\(prefix).conv1.weight", from: &source, dtype: dtype)
        )
        flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).conv1.conv.bias"] = try fetch(
          "\(prefix).conv1.bias", from: &source, dtype: dtype)
        flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).conv2.conv.weight"] = try convertConv3D(
          fetch("\(prefix).conv2.weight", from: &source, dtype: dtype)
        )
        flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).conv2.conv.bias"] = try fetch(
          "\(prefix).conv2.bias", from: &source, dtype: dtype)

        let skipKey = "\(prefix).conv_shortcut.weight"
        if let skipWeight = source.removeValue(forKey: skipKey) {
          let converted = convertConv3D(skipWeight)
          flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).skip.conv.weight"] =
            dtypeAdjusted(converted, dtype: dtype)
          if let skipBias = source.removeValue(forKey: "\(prefix).conv_shortcut.bias") {
            flat["decoder.up_blocks.\(blockIndex).resnets.\(resnetIndex).skip.conv.bias"] =
              dtypeAdjusted(skipBias, dtype: dtype)
          }
        }
      }

      let upsamplerKey = "decoder.up_blocks.\(blockIndex).upsamplers"
      if source.keys.contains(where: { $0.hasPrefix(upsamplerKey) }) {
        flat["decoder.up_blocks.\(blockIndex).upsamplers.0.resample.weight"] = try convertConv2D(
          fetch("\(upsamplerKey).0.resample.1.weight", from: &source, dtype: dtype)
        )
        flat["decoder.up_blocks.\(blockIndex).upsamplers.0.resample.bias"] = try fetch(
          "\(upsamplerKey).0.resample.1.bias", from: &source, dtype: dtype)

        if let timeWeight = source.removeValue(forKey: "\(upsamplerKey).0.time_conv.weight") {
          flat["decoder.up_blocks.\(blockIndex).upsamplers.0.time_conv.conv.weight"] =
            dtypeAdjusted(convertConv3D(timeWeight), dtype: dtype)
          if let timeBias = source.removeValue(forKey: "\(upsamplerKey).0.time_conv.bias") {
            flat["decoder.up_blocks.\(blockIndex).upsamplers.0.time_conv.conv.bias"] =
              dtypeAdjusted(timeBias, dtype: dtype)
          }
        }
      }
    }

    flat["decoder.norm_out.weight"] = try fetchNormTensor(
      "decoder.norm_out", from: &source, dtype: dtype, images: false
    )
    flat["decoder.conv_out.conv.weight"] = try convertConv3D(
      fetch("decoder.conv_out.weight", from: &source, dtype: dtype)
    )
    flat["decoder.conv_out.conv.bias"] = try fetch("decoder.conv_out.bias", from: &source, dtype: dtype)

    // Encoder ----------------------------------------------------------------

    flat["encoder.conv_in.conv.weight"] = try convertConv3D(
      fetch("encoder.conv_in.weight", from: &source, dtype: dtype)
    )
    flat["encoder.conv_in.conv.bias"] = try fetch("encoder.conv_in.bias", from: &source, dtype: dtype)

    // Flattened encoder.down_blocks indices are structured as:
    //   resnets: [0,1], downsample: 2, resnets: [3,4], downsample: 5, resnets: [6,7], downsample: 8, resnets: [9,10]
    let encoderStageResnetIndices: [[Int]] = [
      [0, 1],
      [3, 4],
      [6, 7],
      [9, 10]
    ]
    let encoderDownsamplerIndices: [Int?] = [2, 5, 8, nil]

    for (stageIndex, resnetIndices) in encoderStageResnetIndices.enumerated() {
      let modulePrefix = "encoder.down_blocks.\(stageIndex)"
      for (localIndex, flatIndex) in resnetIndices.enumerated() {
        let prefix = "encoder.down_blocks.\(flatIndex)"
        flat["\(modulePrefix).resnets.\(localIndex).norm1.weight"] = try fetchNormTensor(
          "\(prefix).norm1", from: &source, dtype: dtype, images: false
        )
        flat["\(modulePrefix).resnets.\(localIndex).norm2.weight"] = try fetchNormTensor(
          "\(prefix).norm2", from: &source, dtype: dtype, images: false
        )
        flat["\(modulePrefix).resnets.\(localIndex).conv1.conv.weight"] = try convertConv3D(
          fetch("\(prefix).conv1.weight", from: &source, dtype: dtype)
        )
        flat["\(modulePrefix).resnets.\(localIndex).conv1.conv.bias"] = try fetch(
          "\(prefix).conv1.bias", from: &source, dtype: dtype)
        flat["\(modulePrefix).resnets.\(localIndex).conv2.conv.weight"] = try convertConv3D(
          fetch("\(prefix).conv2.weight", from: &source, dtype: dtype)
        )
        flat["\(modulePrefix).resnets.\(localIndex).conv2.conv.bias"] = try fetch(
          "\(prefix).conv2.bias", from: &source, dtype: dtype)

        let skipKey = "\(prefix).conv_shortcut.weight"
        if let skipWeight = source.removeValue(forKey: skipKey) {
          flat["\(modulePrefix).resnets.\(localIndex).skip.conv.weight"] =
            dtypeAdjusted(convertConv3D(skipWeight), dtype: dtype)
          if let skipBias = source.removeValue(forKey: "\(prefix).conv_shortcut.bias") {
            flat["\(modulePrefix).resnets.\(localIndex).skip.conv.bias"] =
              dtypeAdjusted(skipBias, dtype: dtype)
          }
        }
      }

      if let downsampleIndex = encoderDownsamplerIndices[stageIndex] {
        let base = "encoder.down_blocks.\(downsampleIndex)"
        flat["\(modulePrefix).downsamplers.0.resample.weight"] = try convertConv2D(
          fetch("\(base).resample.1.weight", from: &source, dtype: dtype)
        )
        flat["\(modulePrefix).downsamplers.0.resample.bias"] = try fetch(
          "\(base).resample.1.bias", from: &source, dtype: dtype)

        if let timeWeight = source.removeValue(forKey: "\(base).time_conv.weight") {
          flat["\(modulePrefix).downsamplers.0.time_conv.conv.weight"] =
            dtypeAdjusted(convertConv3D(timeWeight), dtype: dtype)
          if let timeBias = source.removeValue(forKey: "\(base).time_conv.bias") {
            flat["\(modulePrefix).downsamplers.0.time_conv.conv.bias"] =
              dtypeAdjusted(timeBias, dtype: dtype)
          }
        }
      }
    }

    let encoderMidResnets = collectNestedIndices(in: source.keys, components: ["encoder", "mid_block", "resnets"])
    for resIndex in encoderMidResnets.sorted() {
      let prefix = "encoder.mid_block.resnets.\(resIndex)"
      flat["encoder.mid_block.resnets.\(resIndex).norm1.weight"] = try fetchNormTensor(
        "\(prefix).norm1", from: &source, dtype: dtype, images: false
      )
      flat["encoder.mid_block.resnets.\(resIndex).norm2.weight"] = try fetchNormTensor(
        "\(prefix).norm2", from: &source, dtype: dtype, images: false
      )
      flat["encoder.mid_block.resnets.\(resIndex).conv1.conv.weight"] = try convertConv3D(
        fetch("\(prefix).conv1.weight", from: &source, dtype: dtype)
      )
      flat["encoder.mid_block.resnets.\(resIndex).conv1.conv.bias"] = try fetch(
        "\(prefix).conv1.bias", from: &source, dtype: dtype)
      flat["encoder.mid_block.resnets.\(resIndex).conv2.conv.weight"] = try convertConv3D(
        fetch("\(prefix).conv2.weight", from: &source, dtype: dtype)
      )
      flat["encoder.mid_block.resnets.\(resIndex).conv2.conv.bias"] = try fetch(
        "\(prefix).conv2.bias", from: &source, dtype: dtype)
    }

    let encoderAttn = collectNestedIndices(in: source.keys, components: ["encoder", "mid_block", "attentions"])
    for index in encoderAttn.sorted() {
      let prefix = "encoder.mid_block.attentions.\(index)"
      flat["encoder.mid_block.attentions.\(index).norm.weight"] = try fetchNormTensor(
        "\(prefix).norm", from: &source, dtype: dtype, images: true
      )
      flat["encoder.mid_block.attentions.\(index).to_qkv.weight"] = try convertConv2D(
        fetch("\(prefix).to_qkv.weight", from: &source, dtype: dtype)
      )
      flat["encoder.mid_block.attentions.\(index).to_qkv.bias"] = try fetch(
        "\(prefix).to_qkv.bias", from: &source, dtype: dtype)
      flat["encoder.mid_block.attentions.\(index).proj.weight"] = try convertConv2D(
        fetch("\(prefix).proj.weight", from: &source, dtype: dtype)
      )
      flat["encoder.mid_block.attentions.\(index).proj.bias"] = try fetch(
        "\(prefix).proj.bias", from: &source, dtype: dtype)
    }

    flat["encoder.norm_out.weight"] = try fetchNormTensor(
      "encoder.norm_out", from: &source, dtype: dtype, images: false
    )
    flat["encoder.conv_out.conv.weight"] = try convertConv3D(
      fetch("encoder.conv_out.weight", from: &source, dtype: dtype)
    )
    flat["encoder.conv_out.conv.bias"] = try fetch("encoder.conv_out.bias", from: &source, dtype: dtype)

    // Latent projections ------------------------------------------------------

    flat["quant_conv.conv.weight"] = try convertConv3D(
      fetch("quant_conv.weight", from: &source, dtype: dtype)
    )
    flat["quant_conv.conv.bias"] = try fetch("quant_conv.bias", from: &source, dtype: dtype)

    flat["post_quant_conv.conv.weight"] = try convertConv3D(
      fetch("post_quant_conv.weight", from: &source, dtype: dtype)
    )
    flat["post_quant_conv.conv.bias"] = try fetch("post_quant_conv.bias", from: &source, dtype: dtype)

    return ModuleParameters.unflattened(flat)
  }

  private static func filteredTensors(_ tensors: [String: MLXArray]) -> [String: MLXArray] {
    tensors.filter { key, _ in
      guard !key.hasPrefix("lm_head") else { return false }
      guard !key.hasPrefix("visual.") else { return false }
      return true
    }
  }

  private static func assignEmbedding(
    flat: inout [String: MLXArray],
    source: inout [String: MLXArray],
    modulePath: String,
    tensorBase: String,
    dtype: DType?,
    quantization: QwenQuantizationPlan?
  ) throws {
    let quantized = isQuantizedLayer(tensorBase: tensorBase, source: source, quantization: quantization)
    flat["\(modulePath).weight"] = try fetch(
      "\(tensorBase).weight",
      from: &source,
      dtype: dtype,
      convert: !quantized
    )
    if quantized {
      try assignQuantizationBuffers(
        flat: &flat,
        source: &source,
        modulePath: modulePath,
        tensorBase: tensorBase,
        dtype: dtype
      )
    }
  }

  private static func assignLinear(
    flat: inout [String: MLXArray],
    source: inout [String: MLXArray],
    modulePath: String,
    tensorBase: String,
    hasBias: Bool,
    dtype: DType?,
    quantization: QwenQuantizationPlan?
  ) throws {
    LinearLayerRegistry.record(tensorBase)
    let quantized = isQuantizedLayer(tensorBase: tensorBase, source: source, quantization: quantization)
    flat["\(modulePath).weight"] = try fetch(
      "\(tensorBase).weight",
      from: &source,
      dtype: dtype,
      convert: !quantized
    )
    if hasBias {
      flat["\(modulePath).bias"] = try fetch("\(tensorBase).bias", from: &source, dtype: dtype)
    }
    if quantized {
      try assignQuantizationBuffers(
        flat: &flat,
        source: &source,
        modulePath: modulePath,
        tensorBase: tensorBase,
        dtype: dtype
      )
    }
  }

  private static func isQuantizedLayer(
    tensorBase: String,
    source: [String: MLXArray],
    quantization: QwenQuantizationPlan?
  ) -> Bool {
    guard let quantization,
          quantization.quantization(for: tensorBase) != nil,
          source["\(tensorBase).scales"] != nil else {
      return false
    }
    return true
  }

  private static func assignQuantizationBuffers(
    flat: inout [String: MLXArray],
    source: inout [String: MLXArray],
    modulePath: String,
    tensorBase: String,
    dtype: DType?
  ) throws {
    flat["\(modulePath).scales"] = try fetch("\(tensorBase).scales", from: &source, dtype: dtype)
    if source["\(tensorBase).biases"] != nil {
      flat["\(modulePath).biases"] = try fetch("\(tensorBase).biases", from: &source, dtype: dtype)
    }
  }

  private static func fetch(
    _ name: String,
    from tensors: inout [String: MLXArray],
    dtype: DType?,
    convert: Bool = true
  ) throws -> MLXArray {
    guard var tensor = tensors.removeValue(forKey: name) else {
      throw WeightsMappingError.missingTensor(name)
    }
    if convert, let dtype, tensor.dtype != dtype {
      tensor = tensor.asType(dtype)
    }
    return tensor
  }

  private static func collectLayerIndices(from keys: Dictionary<String, MLXArray>.Keys) -> [Int] {
    var indices = Set<Int>()
    for key in keys {
      guard key.hasPrefix("model.layers.") else { continue }
      let components = key.split(separator: ".")
      if components.count > 2, let value = Int(components[2]) {
        indices.insert(value)
      }
    }
    return indices.sorted()
  }

  private static func convertConv2D(_ tensor: MLXArray) -> MLXArray {
    guard tensor.ndim == 4 else { return tensor }
    return tensor.transposed(0, 2, 3, 1)
  }

  private static func convertConv3D(_ tensor: MLXArray) -> MLXArray {
    guard tensor.ndim == 5 else { return tensor }
    return tensor.transposed(0, 2, 3, 4, 1)
  }

  private static func dtypeAdjusted(_ tensor: MLXArray, dtype: DType?) -> MLXArray {
    guard let dtype, tensor.dtype != dtype else { return tensor }
    return tensor.asType(dtype)
  }

  private static func collectNestedIndices(
    in keys: Dictionary<String, MLXArray>.Keys,
    components: [String]
  ) -> Set<Int> {
    var values = Set<Int>()
    for key in keys {
      let parts = key.split(separator: ".")
      guard parts.count >= components.count + 1 else { continue }
      var matches = true
      for (index, component) in components.enumerated()
      where parts[index] != component[component.startIndex...] {
        matches = false
        break
      }
      if matches, let value = Int(parts[components.count]) {
        values.insert(value)
      }
    }
    return values
  }
}
