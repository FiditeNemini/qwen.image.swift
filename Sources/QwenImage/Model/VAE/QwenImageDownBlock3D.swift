import Foundation
import MLX
import MLXNN

final class QwenImageDownBlock3D: Module {
  @ModuleInfo(key: "resnets") var resnets: [QwenImageResBlock3D]
  @ModuleInfo(key: "downsamplers") var downsamplers: [QwenImageResample3D]

  init(
    inChannels: Int,
    outChannels: Int,
    numberOfResBlocks: Int,
    downsampleMode: QwenImageResampleMode?
  ) {
    var blocks: [QwenImageResBlock3D] = []
    var currentChannels = inChannels
    for _ in 0..<numberOfResBlocks {
      blocks.append(QwenImageResBlock3D(inChannels: currentChannels, outChannels: outChannels))
      currentChannels = outChannels
    }
    self._resnets.wrappedValue = blocks

    if let mode = downsampleMode {
      self._downsamplers.wrappedValue = [QwenImageResample3D(channels: outChannels, mode: mode)]
    } else {
      self._downsamplers.wrappedValue = []
    }

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if let downsampler = downsamplers.first {
      hidden = downsampler(hidden)
    }
    return hidden
  }
}
