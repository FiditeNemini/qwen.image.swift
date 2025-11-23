import Foundation
import MLX
import MLXNN

final class QwenTransformerAttention: Module {
  let dimension: Int
  let numberOfHeads: Int
  let headDimension: Int
  var quantizationSpec: QwenQuantizationSpec?

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear

  @ModuleInfo(key: "add_q_proj") var addQProj: Linear
  @ModuleInfo(key: "add_k_proj") var addKProj: Linear
  @ModuleInfo(key: "add_v_proj") var addVProj: Linear

  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm
  @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
  @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

  @ModuleInfo(key: "attn_to_out") var attnToOut: Linear
  @ModuleInfo(key: "to_add_out") var toAddOut: Linear

  init(dim: Int, numHeads: Int, headDim: Int) {
    self.dimension = dim
    self.numberOfHeads = numHeads
    self.headDimension = headDim

    self._toQ.wrappedValue = Linear(dim, dim)
    self._toK.wrappedValue = Linear(dim, dim)
    self._toV.wrappedValue = Linear(dim, dim)

    self._addQProj.wrappedValue = Linear(dim, dim)
    self._addKProj.wrappedValue = Linear(dim, dim)
    self._addVProj.wrappedValue = Linear(dim, dim)

    self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    self._normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    self._normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)

    self._attnToOut.wrappedValue = Linear(dim, dim)
    self._toAddOut.wrappedValue = Linear(dim, dim)
  }

  func callAsFunction(
    imageModulated: MLXArray,
    textModulated: MLXArray,
    encoderHiddenStatesMask: MLXArray?,
    imageRotaryEmbeddings: (MLXArray, MLXArray),
    additiveMask: MLXArray?
  ) -> (MLXArray, MLXArray) {
    let (imgQ, imgK, imgV) = AttentionUtils.processQKV(
      hiddenStates: imageModulated,
      toQ: toQ,
      toK: toK,
      toV: toV,
      normQ: normQ,
      normK: normK,
      numHeads: numberOfHeads,
      headDim: headDimension
    )

    let (txtQ, txtK, txtV) = AttentionUtils.processQKV(
      hiddenStates: textModulated,
      toQ: addQProj,
      toK: addKProj,
      toV: addVProj,
      normQ: normAddedQ,
      normK: normAddedK,
      numHeads: numberOfHeads,
      headDim: headDimension
    )

    var jointQ = MLX.concatenated([txtQ, imgQ], axis: 2)
    var jointK = MLX.concatenated([txtK, imgK], axis: 2)
    let jointV = MLX.concatenated([txtV, imgV], axis: 2)

    let (rotImg, rotTxt) = imageRotaryEmbeddings
    let textSequenceLengthQ = txtQ.dim(2)

    (jointQ, jointK) = QwenTransformerAttention.applyRotaryEmbeddingsJoint(
      jointQ: jointQ,
      jointK: jointK,
      textSequenceLength: textSequenceLengthQ,
      imageRotaryEmbeddings: (rotImg, rotTxt)
    )

    let attentionMask = additiveMask ?? AttentionUtils.convertKeyPaddingMaskToAdditiveMask(
      mask: encoderHiddenStatesMask,
      jointSequenceLength: jointQ.dim(2),
      textSequenceLength: txtQ.dim(2),
      targetDType: jointQ.dtype
    )

    let hiddenStates: MLXArray
    if let spec = quantizationSpec {
      hiddenStates = computeQuantizedAttention(
        jointQ: jointQ,
        jointK: jointK,
        jointV: jointV,
        mask: attentionMask,
        spec: spec
      )
    } else {
      hiddenStates = AttentionUtils.computeAttention(
        query: jointQ,
        key: jointK,
        value: jointV,
        mask: attentionMask
      )
    }

    let textSequenceLength = textModulated.dim(1)
    let textAttentionOutput = hiddenStates[0..., 0..<textSequenceLength, 0...]
    let imageAttentionOutput = hiddenStates[0..., textSequenceLength..., 0...]

    let projectedImage = attnToOut(imageAttentionOutput)
    let projectedText = toAddOut(textAttentionOutput)

    return (projectedImage, projectedText)
  }

  private static func applyRotaryEmbeddingsJoint(
    jointQ: MLXArray,
    jointK: MLXArray,
    textSequenceLength: Int,
    imageRotaryEmbeddings: (MLXArray, MLXArray)
  ) -> (MLXArray, MLXArray) {
    let (imageRotary, textRotary) = imageRotaryEmbeddings

    let imageQuery = jointQ[0..., 0..., textSequenceLength..., 0...]
    let textQuery = jointQ[0..., 0..., ..<textSequenceLength, 0...]
    let imageKey = jointK[0..., 0..., textSequenceLength..., 0...]
    let textKey = jointK[0..., 0..., ..<textSequenceLength, 0...]

    let imageCos = imageRotary[.ellipsis, 0, 0]
      .reshaped(imageRotary.dim(2), imageRotary.dim(3))
    let imageSin = imageRotary[.ellipsis, 1, 0]
      .reshaped(imageRotary.dim(2), imageRotary.dim(3))
    let textCos = textRotary[.ellipsis, 0, 0]
      .reshaped(textRotary.dim(2), textRotary.dim(3))
    let textSin = textRotary[.ellipsis, 1, 0]
      .reshaped(textRotary.dim(2), textRotary.dim(3))

    var imgQ = imageQuery.transposed(0, 2, 1, 3)
    var imgK = imageKey.transposed(0, 2, 1, 3)
    var txtQ = textQuery.transposed(0, 2, 1, 3)
    var txtK = textKey.transposed(0, 2, 1, 3)

    (imgQ, imgK) = AttentionUtils.applyRoPEBSHD(
      query: imgQ,
      key: imgK,
      cos: imageCos,
      sin: imageSin
    )
    (txtQ, txtK) = AttentionUtils.applyRoPEBSHD(
      query: txtQ,
      key: txtK,
      cos: textCos,
      sin: textSin
    )

    imgQ = imgQ.transposed(0, 2, 1, 3)
    imgK = imgK.transposed(0, 2, 1, 3)
    txtQ = txtQ.transposed(0, 2, 1, 3)
    txtK = txtK.transposed(0, 2, 1, 3)

    let jointQuery = MLX.concatenated([txtQ, imgQ], axis: 2)
    let jointKey = MLX.concatenated([txtK, imgK], axis: 2)
    return (jointQuery, jointKey)
  }

  private func computeQuantizedAttention(
    jointQ: MLXArray,
    jointK: MLXArray,
    jointV: MLXArray,
    mask: MLXArray?,
    spec: QwenQuantizationSpec
  ) -> MLXArray {
    let scale = Float(1.0) / sqrt(Float(headDimension))
    let maskMode = mask.map { MLXFast.ScaledDotProductAttentionMaskMode.array($0) } ?? .none

    let qTensor = jointQ.asType(.float32)
    let kTensor = jointK.asType(.float32)
    let vTensor = jointV.asType(.float32)

    let quantizedKeys = MLX.quantized(
      kTensor,
      groupSize: spec.groupSize,
      bits: spec.bits,
      mode: spec.mode
    )
    let quantizedValues = MLX.quantized(
      vTensor,
      groupSize: spec.groupSize,
      bits: spec.bits,
      mode: spec.mode
    )

    var context = AttentionUtils.quantizedScaledDotProductAttention(
      queries: qTensor,
      quantizedKeys: quantizedKeys,
      quantizedValues: quantizedValues,
      scale: scale,
      mask: maskMode,
      groupSize: spec.groupSize,
      bits: spec.bits,
      mode: spec.mode
    )
    context = context.transposed(0, 2, 1, 3)
    context = context.reshaped(
      context.dim(0),
      context.dim(1),
      context.dim(2) * context.dim(3)
    )
    return context.asType(jointQ.dtype)
  }
}
