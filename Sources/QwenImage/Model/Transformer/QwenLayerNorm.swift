import Foundation
import MLX
import MLXNN

final class QwenLayerNorm: Module {
  @ModuleInfo(key: "mod_linear") var modLinear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(dim: Int) {
    self._modLinear.wrappedValue = Linear(dim, 6 * dim)
    self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    textEmbeddings: MLXArray
  ) -> (MLXArray, MLXArray, MLXArray) {
    let modParams = modLinear(MLXNN.silu(textEmbeddings))
    let outputs = modParams.split(parts: 2, axis: -1)
    let mod1 = outputs[0]
    let mod2 = outputs[1]

    var normed = norm(hiddenStates)
    let stage1 = mod1.split(parts: 3, axis: -1)
    let shift1 = MLX.expandedDimensions(stage1[0], axis: 1)
    let scale1 = MLX.expandedDimensions(stage1[1], axis: 1)
    let gate1 = stage1[2]

    normed = normed * (1 + scale1) + shift1

    return (normed, gate1, mod2)
  }
}
