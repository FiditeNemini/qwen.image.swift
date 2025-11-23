import Foundation
import MLX
import MLXNN

public final class QwenVAE: Module {
  static let latentsMean: MLXArray = {
    let values: [Float] = [
      -0.7571, -0.7089, -0.9113, 0.1075,
      -0.1745, 0.9653, -0.1517, 1.5508,
      0.4134, -0.0715, 0.5517, -0.3632,
      -0.1922, -0.9497, 0.2503, -0.2921
    ]
    return MLXArray(values, [1, 16, 1, 1, 1])
  }()

  static let latentsStd: MLXArray = {
    let values: [Float] = [
      2.8184, 1.4541, 2.3275, 2.6558,
      1.2196, 1.7708, 2.6052, 2.0743,
      3.2687, 2.1526, 2.8652, 1.5579,
      1.6382, 1.1253, 2.8251, 1.916
    ]
    return MLXArray(values, [1, 16, 1, 1, 1])
  }()

  @ModuleInfo(key: "encoder") var encoder: QwenImageEncoder3D
  @ModuleInfo(key: "decoder") var decoder: QwenImageDecoder3D
  @ModuleInfo(key: "post_quant_conv") var postQuantConv: QwenImageCausalConv3D
  @ModuleInfo(key: "quant_conv") var quantConv: QwenImageCausalConv3D

  public override init() {
    self._encoder.wrappedValue = QwenImageEncoder3D()
    self._decoder.wrappedValue = QwenImageDecoder3D()
    self._postQuantConv.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 16,
      outputChannels: 16,
      kernelSize: (1, 1, 1),
      stride: (1, 1, 1),
      padding: (0, 0, 0)
    )
    self._quantConv.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 32,
      outputChannels: 32,
      kernelSize: (1, 1, 1),
      stride: (1, 1, 1),
      padding: (0, 0, 0)
    )
    super.init()
  }

  public func decode(_ latents: MLXArray) -> MLXArray {
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let height = latents.dim(2)
    let width = latents.dim(3)

    var hidden = latents
    hidden = hidden.reshaped(batch, channels, 1, height, width)
    hidden = hidden * Self.latentsStd.asType(hidden.dtype) + Self.latentsMean.asType(hidden.dtype)
    hidden = postQuantConv(hidden)
    hidden = decoder(hidden)
    return hidden[0..., 0..., 0, 0..., 0...]
  }

  public func encode(_ images: MLXArray) -> MLXArray {
    let result = encodeWithIntermediates(images)
    return result.latents
  }

  public func encodeWithIntermediates(_ images: MLXArray) -> (latents: MLXArray, encoderHidden: MLXArray, quantHidden: MLXArray) {
    precondition(images.ndim == 4, "Expected input in NCHW format")
    let batch = images.dim(0)
    let channels = images.dim(1)
    let height = images.dim(2)
    let width = images.dim(3)

    var reshaped = images
    reshaped = reshaped.reshaped(batch, channels, 1, height, width)
    let encoderHidden = encoder(reshaped)
    let quantHidden = quantConv(encoderHidden)

    var selected = quantHidden[0..., 0..<16, 0, 0..., 0...]

    let mean = Self.latentsMean[0..., 0..., 0, 0..., 0...].asType(quantHidden.dtype)
    let std = Self.latentsStd[0..., 0..., 0, 0..., 0...].asType(quantHidden.dtype)

    let normalized = (selected - mean) / std
    return (normalized, encoderHidden, quantHidden)
  }
}
