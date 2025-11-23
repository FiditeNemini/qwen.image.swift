import Foundation
import MLX
import MLXNN

final class QwenImageAttentionBlock3D: Module {
  @ModuleInfo(key: "norm") var norm: QwenImageRMSNorm
  @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
  @ModuleInfo(key: "proj") var projection: Conv2d

  init(channels: Int) {
    self._norm.wrappedValue = QwenImageRMSNorm(channels: channels, images: true)
    self._toQKV.wrappedValue = Conv2d(
      inputChannels: channels,
      outputChannels: channels * 3,
      kernelSize: 1,
      stride: 1,
      padding: 0
    )
    self._projection.wrappedValue = Conv2d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 1,
      stride: 1,
      padding: 0
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let identity = x
    let shape = x.shape
    let batch = shape[0]
    let channels = shape[1]
    let time = shape[2]
    let height = shape[3]
    let width = shape[4]

    var hidden = x.transposed(0, 2, 1, 3, 4) // B, T, C, H, W
    hidden = hidden.reshaped(batch * time, channels, height, width)
    hidden = hidden.expandedDimensions(axis: 2)
    hidden = norm(hidden)
    hidden = hidden.squeezed(axis: 2)
    hidden = hidden.transposed(0, 2, 3, 1) // NHWC

    var qkv = toQKV(hidden)
    qkv = qkv.reshaped(batch * time, height * width, channels * 3)
    let splits = qkv.split(parts: 3, axis: -1)
    var q = splits[0]
    var k = splits[1]
    var v = splits[2]

    q = q.reshaped(batch * time, height * width, channels).transposed(0, 2, 1)
    k = k.reshaped(batch * time, height * width, channels).transposed(0, 2, 1)
    v = v.reshaped(batch * time, height * width, channels).transposed(0, 2, 1)

    var attnWeights = MLX.matmul(q.transposed(0, 2, 1), k)
    let scale = MLXArray(sqrtf(Float(channels)))
    attnWeights = attnWeights / scale
    attnWeights = MLX.softmax(attnWeights, axis: -1)

    var attnOutput = MLX.matmul(attnWeights, v.transposed(0, 2, 1))
    attnOutput = attnOutput.reshaped(batch * time, height, width, channels)
    attnOutput = projection(attnOutput)

    attnOutput = attnOutput.transposed(0, 3, 1, 2)
    attnOutput = attnOutput.reshaped(batch, time, channels, height, width)
    attnOutput = attnOutput.transposed(0, 2, 1, 3, 4).asType(identity.dtype)

    return identity + attnOutput
  }
}
