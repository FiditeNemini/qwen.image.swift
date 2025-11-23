import Foundation
import MLX
import MLXNN

final class QwenTimestepEmbedding: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  init(projectionDim: Int, innerDim: Int) {
    self._linear1.wrappedValue = Linear(projectionDim, innerDim)
    self._linear2.wrappedValue = Linear(innerDim, innerDim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = MLXNN.silu(linear1(x))
    hidden = linear2(hidden)
    return hidden
  }
}
