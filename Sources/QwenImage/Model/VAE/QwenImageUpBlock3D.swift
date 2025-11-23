import Foundation
import MLX
import MLXNN

final class QwenImageUpBlock3D: Module {
  @ModuleInfo(key: "resnets") var resnets: [QwenImageResBlock3D]
  @ModuleInfo(key: "upsamplers") var upsamplers: [QwenImageResample3D]

  init(inChannels: Int, outChannels: Int, numberOfResBlocks: Int, upsampleMode: QwenImageResampleMode?) {
    var blocks: [QwenImageResBlock3D] = []
    for index in 0...numberOfResBlocks {
      let isFirst = index == 0
      let inputChannels = isFirst ? inChannels : outChannels
      blocks.append(QwenImageResBlock3D(inChannels: inputChannels, outChannels: outChannels))
    }
    self._resnets.wrappedValue = blocks
    if let mode = upsampleMode {
      self._upsamplers.wrappedValue = [QwenImageResample3D(channels: outChannels, mode: mode)]
    } else {
      self._upsamplers.wrappedValue = []
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if let upsampler = upsamplers.first {
      hidden = upsampler(hidden)
    }
    return hidden
  }
}
