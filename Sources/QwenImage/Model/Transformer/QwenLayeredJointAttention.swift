import Foundation
import MLX
import MLXFast
import MLXNN

public final class QwenLayeredJointAttentionV2: Module {
  let numHeads: Int
  let headDim: Int
  let innerDim: Int

  @ModuleInfo(key: "to_q") public var toQ: Linear
  @ModuleInfo(key: "to_k") public var toK: Linear
  @ModuleInfo(key: "to_v") public var toV: Linear
  @ModuleInfo(key: "to_out") public var toOut: [Linear]

  @ModuleInfo(key: "add_q_proj") var addQProj: Linear
  @ModuleInfo(key: "add_k_proj") var addKProj: Linear
  @ModuleInfo(key: "add_v_proj") var addVProj: Linear
  @ModuleInfo(key: "to_add_out") var toAddOut: Linear

  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm
  @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
  @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm

  public init(dim: Int, numHeads: Int, headDim: Int) {
    self.numHeads = numHeads
    self.headDim = headDim
    self.innerDim = numHeads * headDim

    self._toQ.wrappedValue = Linear(dim, innerDim, bias: true)
    self._toK.wrappedValue = Linear(dim, innerDim, bias: true)
    self._toV.wrappedValue = Linear(dim, innerDim, bias: true)
    self._toOut.wrappedValue = [Linear(innerDim, dim, bias: true)]

    self._addQProj.wrappedValue = Linear(dim, innerDim, bias: true)
    self._addKProj.wrappedValue = Linear(dim, innerDim, bias: true)
    self._addVProj.wrappedValue = Linear(dim, innerDim, bias: true)
    self._toAddOut.wrappedValue = Linear(innerDim, dim, bias: true)

    self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    self._normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    self._normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
  }

  public func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    imageRotaryEmb: (vid: MLXArray, txt: MLXArray)
  ) -> (MLXArray, MLXArray) {
    let batch = hiddenStates.dim(0)
    let imageSeqLen = hiddenStates.dim(1)
    let textSeqLen = encoderHiddenStates.dim(1)

    var imgQuery = toQ(hiddenStates)
    var imgKey = toK(hiddenStates)
    var imgValue = toV(hiddenStates)

    var txtQuery = addQProj(encoderHiddenStates)
    var txtKey = addKProj(encoderHiddenStates)
    var txtValue = addVProj(encoderHiddenStates)

    imgQuery = imgQuery.reshaped([batch, imageSeqLen, numHeads, headDim])
    imgKey = imgKey.reshaped([batch, imageSeqLen, numHeads, headDim])
    imgValue = imgValue.reshaped([batch, imageSeqLen, numHeads, headDim])

    txtQuery = txtQuery.reshaped([batch, textSeqLen, numHeads, headDim])
    txtKey = txtKey.reshaped([batch, textSeqLen, numHeads, headDim])
    txtValue = txtValue.reshaped([batch, textSeqLen, numHeads, headDim])

    imgQuery = normQ(imgQuery)
    imgKey = normK(imgKey)
    txtQuery = normAddedQ(txtQuery)
    txtKey = normAddedK(txtKey)

    let (imageFreqs, textFreqs) = imageRotaryEmb
    imgQuery = applyRotaryEmbQwen(imgQuery, freqs: imageFreqs)
    imgKey = applyRotaryEmbQwen(imgKey, freqs: imageFreqs)
    txtQuery = applyRotaryEmbQwen(txtQuery, freqs: textFreqs)
    txtKey = applyRotaryEmbQwen(txtKey, freqs: textFreqs)

    // Text first, then image for joint attention
    var jointQuery = MLX.concatenated([txtQuery, imgQuery], axis: 1)
    var jointKey = MLX.concatenated([txtKey, imgKey], axis: 1)
    let jointValue = MLX.concatenated([txtValue, imgValue], axis: 1)

    jointQuery = jointQuery.transposed(0, 2, 1, 3)
    jointKey = jointKey.transposed(0, 2, 1, 3)
    let jointValueT = jointValue.transposed(0, 2, 1, 3)

    let scale = pow(Float(headDim), -0.5)
    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: jointQuery,
      keys: jointKey,
      values: jointValueT,
      scale: scale,
      mask: nil
    )

    attnOutput = attnOutput.transposed(0, 2, 1, 3)
    attnOutput = attnOutput.reshaped([batch, textSeqLen + imageSeqLen, innerDim])

    let txtAttnOutput = attnOutput[0..., 0..<textSeqLen, 0...]
    let imgAttnOutput = attnOutput[0..., textSeqLen..., 0...]

    let imgOutput = toOut[0](imgAttnOutput)
    let txtOutput = toAddOut(txtAttnOutput)

    return (imgOutput, txtOutput)
  }
}
