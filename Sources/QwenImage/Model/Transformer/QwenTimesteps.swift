import Foundation
import MLX
import MLXNN

final class QwenTimesteps: Module {
  let projectionDim: Int
  let scale: Float

  init(projectionDim: Int = 256, scale: Float = 1000.0) {
    self.projectionDim = projectionDim
    self.scale = scale
  }

  func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
    let halfDim = projectionDim / 2
    let maxPeriod: Float = 10_000.0
    var exponent = -MLX.log(MLXArray(maxPeriod))
    exponent = exponent * MLXArray(0..<halfDim).asType(.float32)
    exponent = exponent / MLXArray(Float(halfDim))
    let freqs = MLX.exp(exponent)

    var emb = timesteps.asType(.float32)[.ellipsis, .newAxis] * freqs[.newAxis]
    emb = emb * MLXArray(scale)
    emb = MLX.concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)
    emb = MLX.concatenated([emb[.ellipsis, halfDim...], emb[.ellipsis, ..<halfDim]], axis: -1)
    return emb
  }
}
