import Foundation
import MLX
import MLXNN

final class QwenImageCausalConv3D: Module {
  @ModuleInfo(key: "conv") private var conv: Conv3d
  private let padding: (Int, Int, Int)
  private let stride: (Int, Int, Int)
  private let kernelSize: (Int, Int, Int)

  init(
    inputChannels: Int,
    outputChannels: Int,
    kernelSize: (Int, Int, Int) = (3, 3, 3),
    stride: (Int, Int, Int) = (1, 1, 1),
    padding: (Int, Int, Int) = (1, 1, 1),
    useBias: Bool = true
  ) {
    self.padding = padding
    self.stride = stride
    self.kernelSize = kernelSize
    self._conv.wrappedValue = Conv3d(
      inputChannels: inputChannels,
      outputChannels: outputChannels,
      kernelSize: .init(kernelSize),
      stride: .init(stride),
      padding: .init(0),
      bias: useBias
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    precondition(x.ndim == 5, "Expected input in NCTHW format")

    var padded = applyCausalPadding(x)
    padded = padded.transposed(0, 2, 3, 4, 1)

    let y = conv(padded)
    return y.transposed(0, 4, 1, 2, 3)
  }

  private func applyCausalPadding(_ input: MLXArray) -> MLXArray {
    let (padT, padH, padW) = padding
    guard padT > 0 || padH > 0 || padW > 0 else { return input }

    let widths: [IntOrPair] = [
      IntOrPair((0, 0)),                // batch
      IntOrPair((0, 0)),                // channels
      IntOrPair((padT * 2, 0)),         // time (causal: pad before only, doubling for compatibility)
      IntOrPair((padH, padH)),          // height
      IntOrPair((padW, padW))           // width
    ]
    return MLX.padded(input, widths: widths)
  }
}
