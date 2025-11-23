import Foundation
import MLX
import MLXNN

final class QwenFeedForward: Module {
  @ModuleInfo(key: "mlp_in") var mlpIn: Linear
  @ModuleInfo(key: "mlp_out") var mlpOut: Linear

  init(dim: Int) {
    self._mlpIn.wrappedValue = Linear(dim, 4 * dim)
    self._mlpOut.wrappedValue = Linear(4 * dim, dim)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var x = mlpIn(hiddenStates)
    x = MLXNN.geluApproximate(x)
    x = mlpOut(x)
    return x
  }
}
