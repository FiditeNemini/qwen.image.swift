import Foundation
import MLX
import MLXNN

final class QwenAdaLayerNormContinuous: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm
  let embeddingDimension: Int

  init(embeddingDimension: Int, conditioningDimension: Int, epsilon: Float = 1e-6) {
    self.embeddingDimension = embeddingDimension
    self._linear.wrappedValue = Linear(conditioningDimension, embeddingDimension * 2)
    self._norm.wrappedValue = LayerNorm(dimensions: embeddingDimension, eps: epsilon, affine: false)
  }

  func callAsFunction(hiddenStates: MLXArray, conditioning: MLXArray) -> MLXArray {
    let conditioningEmbeds = linear(MLXNN.silu(conditioning))
    let scale = conditioningEmbeds[0..., 0..<embeddingDimension]
    let shift = conditioningEmbeds[0..., embeddingDimension..<embeddingDimension * 2]

    var normalized = norm(hiddenStates)
    normalized = normalized * MLX.expandedDimensions(1 + scale, axis: 1)
      + MLX.expandedDimensions(shift, axis: 1)
    return normalized
  }
}
