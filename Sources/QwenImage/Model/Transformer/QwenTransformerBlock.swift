import Foundation
import MLX
import MLXNN

final class QwenTransformerBlock: Module {
  @ModuleInfo(key: "img_norm1") var imageNorm1: QwenLayerNorm
  @ModuleInfo(key: "txt_norm1") var textNorm1: QwenLayerNorm
  @ModuleInfo(key: "attn") var attention: QwenTransformerAttention
  @ModuleInfo(key: "img_norm2") var imageNorm2: LayerNorm
  @ModuleInfo(key: "txt_norm2") var textNorm2: LayerNorm
  @ModuleInfo(key: "img_ff") var imageFeedForward: QwenFeedForward
  @ModuleInfo(key: "txt_ff") var textFeedForward: QwenFeedForward

  init(dimension: Int, numberOfHeads: Int, headDimension: Int) {
    self._imageNorm1.wrappedValue = QwenLayerNorm(dim: dimension)
    self._textNorm1.wrappedValue = QwenLayerNorm(dim: dimension)
    self._attention.wrappedValue = QwenTransformerAttention(
      dim: dimension,
      numHeads: numberOfHeads,
      headDim: headDimension
    )
    self._imageNorm2.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6, affine: false)
    self._textNorm2.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6, affine: false)
    self._imageFeedForward.wrappedValue = QwenFeedForward(dim: dimension)
    self._textFeedForward.wrappedValue = QwenFeedForward(dim: dimension)
  }

  func setAttentionQuantization(_ spec: QwenQuantizationSpec?) {
    attention.quantizationSpec = spec
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray?,
    textEmbeddings: MLXArray,
    imageRotaryEmbeddings: (MLXArray, MLXArray),
    additiveMask: MLXArray?
  ) -> (MLXArray, MLXArray) {
    let (imageModulated, imageGate, imageMod2) = imageNorm1(
      hiddenStates,
      textEmbeddings: textEmbeddings
    )
    let (textModulated, textGate, textMod2) = textNorm1(
      encoderHiddenStates,
      textEmbeddings: textEmbeddings
    )

    let (imageAttentionOutput, textAttentionOutput) = attention(
      imageModulated: imageModulated,
      textModulated: textModulated,
      encoderHiddenStatesMask: encoderHiddenStatesMask,
      imageRotaryEmbeddings: imageRotaryEmbeddings,
      additiveMask: additiveMask
    )

    let updatedHiddenStates = QwenTransformerBlock.applyResidualAndFeedForward(
      hiddenStates: hiddenStates,
      output: imageAttentionOutput,
      gateAttention: imageGate,
      modParameters: imageMod2,
      normalization: imageNorm2,
      feedForward: imageFeedForward
    )

    let updatedEncoderHiddenStates = QwenTransformerBlock.applyResidualAndFeedForward(
      hiddenStates: encoderHiddenStates,
      output: textAttentionOutput,
      gateAttention: textGate,
      modParameters: textMod2,
      normalization: textNorm2,
      feedForward: textFeedForward
    )

    return (updatedEncoderHiddenStates, updatedHiddenStates)
  }

  private static func applyResidualAndFeedForward(
    hiddenStates: MLXArray,
    output: MLXArray,
    gateAttention: MLXArray,
    modParameters: MLXArray,
    normalization: LayerNorm,
    feedForward: QwenFeedForward
  ) -> MLXArray {
    let gatedOutput = hiddenStates + MLX.expandedDimensions(gateAttention, axis: 1) * output

    let modParts = modParameters.split(parts: 3, axis: -1)
    let shift = MLX.expandedDimensions(modParts[0], axis: 1)
    let scale = MLX.expandedDimensions(modParts[1], axis: 1)
    let gateFF = MLX.expandedDimensions(modParts[2], axis: 1)

    let normalized = normalization(gatedOutput)
    let modulated = normalized * (1 + scale) + shift

    let ffOutput = feedForward(modulated)
    return gatedOutput + gateFF * ffOutput
  }
}
