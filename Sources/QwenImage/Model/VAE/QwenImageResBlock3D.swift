import Foundation
import MLX
import MLXNN

final class QwenImageResBlock3D: Module {
  @ModuleInfo(key: "norm1") var norm1: QwenImageRMSNorm
  @ModuleInfo(key: "conv1") var conv1: QwenImageCausalConv3D
  @ModuleInfo(key: "norm2") var norm2: QwenImageRMSNorm
  @ModuleInfo(key: "conv2") var conv2: QwenImageCausalConv3D
  @ModuleInfo(key: "skip") var skip: QwenImageCausalConv3D?

  init(inChannels: Int, outChannels: Int) {
    self._norm1.wrappedValue = QwenImageRMSNorm(channels: inChannels, images: false)
    self._conv1.wrappedValue = QwenImageCausalConv3D(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: (3, 3, 3),
      stride: (1, 1, 1),
      padding: (1, 1, 1)
    )
    self._norm2.wrappedValue = QwenImageRMSNorm(channels: outChannels, images: false)
    self._conv2.wrappedValue = QwenImageCausalConv3D(
      inputChannels: outChannels,
      outputChannels: outChannels,
      kernelSize: (3, 3, 3),
      stride: (1, 1, 1),
      padding: (1, 1, 1)
    )
    if inChannels != outChannels {
      self._skip.wrappedValue = QwenImageCausalConv3D(
        inputChannels: inChannels,
        outputChannels: outChannels,
        kernelSize: (1, 1, 1),
        stride: (1, 1, 1),
        padding: (0, 0, 0)
      )
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = norm1(x).asType(x.dtype)
    hidden = MLXNN.silu(hidden)
    hidden = conv1(hidden)

    hidden = norm2(hidden).asType(x.dtype)
    hidden = MLXNN.silu(hidden)
    hidden = conv2(hidden)

    var residual = x
    if let skip {
      residual = skip(residual)
    }

    return hidden + residual
  }
}
