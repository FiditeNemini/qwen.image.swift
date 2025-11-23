import Foundation
import MLX
import MLXNN

final class QwenImageMidBlock3D: Module {
  @ModuleInfo(key: "resnets") var resnets: [QwenImageResBlock3D]
  @ModuleInfo(key: "attentions") var attentions: [QwenImageAttentionBlock3D]

  init(channels: Int, attentionLayers: Int = 1) {
    var resBlocks: [QwenImageResBlock3D] = []
    resBlocks.append(QwenImageResBlock3D(inChannels: channels, outChannels: channels))
    for _ in 0..<attentionLayers {
      resBlocks.append(QwenImageResBlock3D(inChannels: channels, outChannels: channels))
    }
    self._resnets.wrappedValue = resBlocks
    self._attentions.wrappedValue = (0..<attentionLayers).map { _ in
      QwenImageAttentionBlock3D(channels: channels)
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = resnets[0](x)
    for index in 0..<attentions.count {
      hidden = attentions[index](hidden)
      hidden = resnets[index + 1](hidden)
    }
    return hidden
  }
}
