import Foundation
import MLX
import MLXNN

enum QwenImageResampleMode: String {
  case upsample3d
  case upsample2d
  case downsample3d
  case downsample2d
}

final class QwenImageResample3D: Module {
  let mode: QwenImageResampleMode
  @ModuleInfo(key: "resample") var resampleConv: Conv2d
  @ModuleInfo(key: "time_conv") var timeConv: QwenImageCausalConv3D?

  init(channels: Int, mode: QwenImageResampleMode) {
    self.mode = mode
    switch mode {
    case .upsample3d:
      self._timeConv.wrappedValue = QwenImageCausalConv3D(
        inputChannels: channels,
        outputChannels: channels * 2,
        kernelSize: (3, 1, 1),
        stride: (1, 1, 1),
        padding: (1, 0, 0)
      )
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: channels,
        outputChannels: channels / 2,
        kernelSize: 3,
        stride: 1,
        padding: 1
      )
    case .upsample2d:
      self._timeConv.wrappedValue = nil
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: channels,
        outputChannels: channels / 2,
        kernelSize: 3,
        stride: 1,
        padding: 1
      )
    case .downsample3d:
      self._timeConv.wrappedValue = QwenImageCausalConv3D(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: (3, 1, 1),
        stride: (2, 1, 1),
        padding: (0, 0, 0)
      )
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: 3,
        stride: 2,
        padding: 0
      )
    case .downsample2d:
      self._timeConv.wrappedValue = nil
      self._resampleConv.wrappedValue = Conv2d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: 3,
        stride: 2,
        padding: 0
      )
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x

    if mode == .downsample3d, let timeConv, hidden.dim(2) >= 3 {
      // For single-frame inputs (T == 1), Diffusers does not apply the temporal conv
      // in downsample3d unless a temporal cache (feat_cache) is provided. Our pipeline
      // operates on single images (T = 1), so we effectively skip this path.
      // This conditional remains conservative to avoid applying a temporal kernel
      // in scenarios we don’t currently support (no temporal cache/chunking).
      hidden = timeConv(hidden)
    }

    let batch = hidden.dim(0)
    let channels = hidden.dim(1)
    let time = hidden.dim(2)
    let height = hidden.dim(3)
    let width = hidden.dim(4)

    hidden = hidden.transposed(0, 2, 1, 3, 4).reshaped(batch * time, channels, height, width)
    hidden = hidden.transposed(0, 2, 3, 1) // NHWC

    if mode == .upsample3d || mode == .upsample2d {
      hidden = QwenImageResample3D.nearestUpsample(hidden, scale: 2)
    }

    if mode == .downsample2d || mode == .downsample3d {
      hidden = MLX.padded(
        hidden,
        widths: [
          .init((0, 0)),
          .init((0, 1)),
          .init((0, 1)),
          .init((0, 0))
        ]
      )
    }

    hidden = resampleConv(hidden)

    var reshaped = hidden.transposed(0, 3, 1, 2)
    let newChannels = reshaped.dim(1)
    let newHeight = reshaped.dim(2)
    let newWidth = reshaped.dim(3)
    reshaped = reshaped.reshaped(batch, time, newChannels, newHeight, newWidth)
    reshaped = reshaped.transposed(0, 2, 1, 3, 4)

    return reshaped
  }

  private static func nearestUpsample(_ input: MLXArray, scale: Int) -> MLXArray {
    precondition(input.ndim == 4)
    let shape = input.shape
    let batch = shape[0]
    let height = shape[1]
    let width = shape[2]
    let channels = shape[3]
    var expanded = input[
      0..., 0..., .newAxis, 0..., .newAxis, 0...
    ]
    expanded = MLX.broadcast(expanded, to: [batch, height, scale, width, scale, channels])
    return expanded.reshaped(batch, height * scale, width * scale, channels)
  }
}
