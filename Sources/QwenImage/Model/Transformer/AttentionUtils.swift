import Foundation
import MLX
import MLXFast
import MLXNN

enum AttentionUtils {

  static func processQKV(
    hiddenStates: MLXArray,
    toQ: Linear,
    toK: Linear,
    toV: Linear,
    normQ: RMSNorm,
    normK: RMSNorm,
    numHeads: Int,
    headDim: Int
  ) -> (MLXArray, MLXArray, MLXArray) {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    var query = toQ(hiddenStates)
    var key = toK(hiddenStates)
    var value = toV(hiddenStates)

    query = query.reshaped(batchSize, seqLen, numHeads, headDim)
      .transposed(0, 2, 1, 3)
    key = key.reshaped(batchSize, seqLen, numHeads, headDim)
      .transposed(0, 2, 1, 3)
    value = value.reshaped(batchSize, seqLen, numHeads, headDim)
      .transposed(0, 2, 1, 3)

    let qType = query.dtype
    let kType = key.dtype
    query = normQ(query).asType(qType)
    key = normK(key).asType(kType)

    return (query, key, value)
  }

  static func computeAttention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    mask: MLXArray?
  ) -> MLXArray {
    let scale = Float(1.0) / sqrt(Float(query.dim(query.ndim - 1)))
    let maskMode = mask.map { MLXFast.ScaledDotProductAttentionMaskMode.array($0) } ?? .none
    var hiddenStates = MLXFast.scaledDotProductAttention(
      queries: query,
      keys: key,
      values: value,
      scale: scale,
      mask: maskMode
    )
    hiddenStates = hiddenStates.transposed(0, 2, 1, 3)
    hiddenStates = hiddenStates.reshaped(
      hiddenStates.dim(0),
      hiddenStates.dim(1),
      hiddenStates.dim(2) * hiddenStates.dim(3)
    )
    return hiddenStates
  }

  // MARK: - Quantized Attention (ported from mlx-swift-lm pattern)

  /// Quantized scaled dot-product attention using MLX quantizedMatmul.
  ///
  /// Expects pre-quantized keys and values in (weights, scales, biases) tuples.
  /// Preserves GQA semantics and mask handling consistent with MLXFast.SDPA.
  ///
  /// Shapes:
  /// - queries: [B, Nq, L, D]
  /// - quantizedKeys.0 (weights): packed quantized tensor compatible with MLX.quantizedMatmul
  /// - quantizedValues.0 (weights): packed quantized tensor compatible with MLX.quantizedMatmul
  ///
  /// Notes:
  /// - This function does not quantize on the fly. Use MLX.quantized(...) beforehand
  ///   or a KV cache that stores quantized K/V.
  /// - When `Nq > Nkv` (GQA), keys/values are broadcast across repeats.
  static func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine
  ) -> MLXArray {
    let B = queries.dim(0)
    let nQHeads = queries.dim(1)
    let L = queries.dim(2)
    let D = queries.dim(3)

    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA by introducing a repeat dimension and broadcasting K/V across it
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
      scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
      qKeys = (
        MLX.expandedDimensions(qKeys.0, axis: -3),
        MLX.expandedDimensions(qKeys.1, axis: -3),
        qKeys.2.map { MLX.expandedDimensions($0, axis: -3) }
      )
      qValues = (
        MLX.expandedDimensions(qValues.0, axis: -3),
        MLX.expandedDimensions(qValues.1, axis: -3),
        qValues.2.map { MLX.expandedDimensions($0, axis: -3) }
      )
    }

    // Compute attention scores using quantized matmul: (Q * scale) @ K^T
    var scores = quantizedMatmul(
      scaledQueries,
      qKeys.0,
      scales: qKeys.1,
      biases: qKeys.2,
      transpose: true,
      groupSize: groupSize,
      bits: bits,
      mode: mode
    )

    // Apply mask
    switch mask {
    case .causal:
      let qL = scores.dim(-2)
      let kL = scores.dim(-1)
      let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
      let kIndices = MLXArray(0 ..< kL)
      let causalMask = greaterEqual(
        MLX.expandedDimensions(qIndices, axis: -1),
        MLX.expandedDimensions(kIndices, axis: -2)
      )
      scores = MLX.where(causalMask, scores, MLXArray(Float.leastNormalMagnitude))

    case .array(let maskArray):
      if maskArray.dtype == .bool {
        scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
      } else {
        scores = scores + maskArray
      }

    case .arrays(let maskArrays):
      if let maskArray = maskArrays.first {
        if maskArray.dtype == .bool {
          scores = MLX.where(maskArray, scores, MLXArray(Float.leastNormalMagnitude))
        } else {
          scores = scores + maskArray
        }
      }

    case .none:
      break
    }

    let attentionWeights = softmax(scores, axis: -1)

    // Compute output using quantized matmul: softmax(scores) @ V
    var output = quantizedMatmul(
      attentionWeights,
      qValues.0,
      scales: qValues.1,
      biases: qValues.2,
      transpose: false,
      groupSize: groupSize,
      bits: bits,
      mode: mode
    )

    // Collapse repeat dimension if GQA
    if nRepeats > 1 {
      output = output.reshaped([B, nQHeads, L, D])
    }
    return output
  }

  static func convertKeyPaddingMaskToAdditiveMask(
    mask: MLXArray?,
    jointSequenceLength: Int,
    textSequenceLength: Int,
    targetDType: DType
  ) -> MLXArray? {
    guard let mask else { return nil }
    let batch = mask.dim(0)
    let imageLength = jointSequenceLength - textSequenceLength
    precondition(imageLength >= 0, "jointSequenceLength must be >= textSequenceLength")
    let imageMask = MLX.ones([batch, imageLength], dtype: targetDType)
    let jointMask = MLX.concatenated(
      [mask.asType(targetDType), imageMask],
      axis: 1
    )
    let one = MLXArray(Float32(1.0), dtype: targetDType)
    let negInf = MLXArray(Float32(-1e9), dtype: targetDType)
    let additive = (one - jointMask) * negInf
    return additive.reshaped(batch, 1, 1, additive.dim(1))
  }

  static func applyRoPE(
    query: MLXArray,
    key: MLXArray,
    freqs: MLXArray
  ) -> (MLXArray, MLXArray) {
    let featureDim = query.dim(query.ndim - 1)
    let ropeShape = Array(query.shape.dropLast() + [featureDim / 2, 1, 2])
    let computeType = freqs.dtype
    let reshapedQuery = query.asType(computeType).reshaped(ropeShape)
    let reshapedKey = key.asType(computeType).reshaped(ropeShape)

    let qOut = freqs[.ellipsis, 0] * reshapedQuery[.ellipsis, 0]
      + freqs[.ellipsis, 1] * reshapedQuery[.ellipsis, 1]
    let kOut = freqs[.ellipsis, 0] * reshapedKey[.ellipsis, 0]
      + freqs[.ellipsis, 1] * reshapedKey[.ellipsis, 1]

    return (
      qOut.reshaped(query.shape).asType(query.dtype),
      kOut.reshaped(key.shape).asType(key.dtype)
    )
  }

  static func applyRoPEBSHD(
    query: MLXArray,
    key: MLXArray,
    cos: MLXArray,
    sin: MLXArray
  ) -> (MLXArray, MLXArray) {
    let computeType = cos.dtype
    let queryTensor = query.asType(computeType)
    let keyTensor = key.asType(computeType)
    let cosPrepared = cos.asType(computeType)
    let sinPrepared = sin.asType(computeType)

    let cosExpanded = cosPrepared.reshaped(1, cosPrepared.dim(0), 1, cosPrepared.dim(1))
    let sinExpanded = sinPrepared.reshaped(1, sinPrepared.dim(0), 1, sinPrepared.dim(1))

    func mix(_ tensor: MLXArray) -> MLXArray {
      let lastDim = tensor.dim(tensor.ndim - 1)
      let reshapeShape = Array(tensor.shape.dropLast() + [lastDim / 2, 2])
      let reshaped = tensor.reshaped(reshapeShape)
      let real = reshaped[.ellipsis, 0]
      let imag = reshaped[.ellipsis, 1]
      let out0 = real * cosExpanded + (-imag) * sinExpanded
      let out1 = imag * cosExpanded + real * sinExpanded
      return MLX.stacked([out0, out1], axis: -1).reshaped(tensor.shape)
    }

    return (
      mix(queryTensor).asType(query.dtype),
      mix(keyTensor).asType(key.dtype)
    )
  }
}
