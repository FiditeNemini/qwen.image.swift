import Foundation
import MLX
import MLXNN

final class QwenImageRMSNorm: Module {
  let eps: Float
  let scale: Float
  let images: Bool
  var weight: MLXArray

  init(channels: Int, eps: Float = 1e-12, images: Bool = true) {
    self.eps = eps
    self.scale = Float(channels).squareRoot()
    self.images = images
    if images {
      self.weight = MLX.ones([channels, 1, 1])
    } else {
      self.weight = MLX.ones([channels, 1, 1, 1])
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    precondition(x.ndim == 4 || x.ndim == 5, "Expected NCHW or NCTHW layout")

    let sumSquares = MLX.sum(x * x, axes: [1], keepDims: true)
    let denom = MLX.maximum(MLX.sqrt(sumSquares), MLXArray(eps, dtype: sumSquares.dtype))
    var normalized = x / denom

    let weightArray: MLXArray
    if x.ndim == 5 {
      weightArray = weight.reshaped(1, weight.dim(0), 1, 1, 1)
    } else if x.ndim == 4 {
      weightArray = weight.reshaped(1, weight.dim(0), 1, 1)
    } else {
      weightArray = weight
    }

    normalized = normalized * weightArray
    normalized = normalized * MLXArray(scale, dtype: normalized.dtype)
    return normalized
  }
}
