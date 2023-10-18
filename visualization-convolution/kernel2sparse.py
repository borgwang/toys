import numpy as np
from PIL import Image


def from_image(infilename):
  img = Image.open(infilename)
  img.load()
  return np.asarray(img, dtype="int32")

def get_padding(inputs, ks, mode="SAME"):
  """
  params: inputs (input array)
  params: ks (kernel size) [p, q]
  return: padding list [n,m,j,k] in different modes
  """
  pad = None
  if mode == "FULL":
    pad = [ks[0] - 1, ks[1] - 1, ks[0] - 1, ks[1] - 1]
  elif mode == "VALID":
    pad = [0, 0, 0, 0]
  elif mode == "SAME":
    pad = [(ks[0] - 1) // 2, (ks[1] - 1) // 2, (ks[0] - 1) // 2, (ks[1] - 1) // 2]
    if ks[0] % 2 == 0:
      pad[2] += 1
    if ks[1] % 2 == 0:
      pad[3] += 1
  else:
    print("Invalid mode")
  return pad

def conv1(kernel, inputs, stride=1, mode="SAME"):
  """将 kernel 展开成一个矩阵"""
  ks = kernel.shape[:2]
  pad = get_padding(inputs, ks, mode=mode)
  padded_inputs = np.pad(inputs, pad_width=((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), mode="constant")

  height, width, _ = inputs.shape
  out_width = int((width + pad[0] + pad[2] - ks[0]) / stride + 1)
  out_height = int((height + pad[1] + pad[3] - ks[1]) / stride + 1)

  rearrange = []
  for row in range(0, padded_inputs.shape[0]-ks[1]+1, stride):
    for col in range(0, padded_inputs.shape[1]-ks[0]+1, stride):
      patch = np.zeros_like(padded_inputs, dtype=np.float32)
      patch[row:row+ks[1], col:col+ks[0], :] = kernel
      rearrange.append(patch.ravel())
  rearrange = np.asarray(rearrange).T
  print(f"expanded kernel shape: {rearrange.shape}")
  print(f"number of zero elements: {(rearrange == 0).sum()}/{np.prod(rearrange.shape)}")
  padded_inputs = padded_inputs.reshape(1, -1)
  return np.matmul(padded_inputs, rearrange).reshape(out_height, out_width)

def conv2(kernel, inputs, stride=1, mode="SAME"):
  """将 inputs 展开成一个矩阵"""
  ks = kernel.shape[:2]
  pad = get_padding(inputs, ks, mode=mode)
  padded_inputs = np.pad(inputs, pad_width=((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), mode="constant")

  height, width, _ = inputs.shape
  out_width = int((width + pad[0] + pad[2] - ks[0]) / stride + 1)
  out_height = int((height + pad[1] + pad[3] - ks[1]) / stride + 1)

  rearrange = []
  for y in range(0, padded_inputs.shape[0]-ks[1]+1, stride):
    for x in range(0, padded_inputs.shape[1]-ks[0]+1, stride):
      patch = padded_inputs[y:y+ks[1], x:x+ks[0], :]
      rearrange.append(patch.ravel())
  rearrange = np.asarray(rearrange).T
  print(f"expanded inputs shape: {rearrange.shape}")
  kernel = kernel.reshape(1, -1)
  return np.matmul(kernel, rearrange).reshape(out_height, out_width)

def test():
  kernel_one_channel = np.array([[0.1, 0.1, 0.1], [0.1, -0.8, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32)
  kernel = np.stack([kernel_one_channel] * 3, axis=2)

  inputs = from_image("./Lenna_test_image.png")
  inputs = inputs[::4, ::4, :]

  print("方法1: 将 kernel 展开成一个矩阵")
  result1 = conv1(kernel, inputs, stride=1)
  print("方法2: 将 inputs 展开成一个矩阵")
  result2 = conv2(kernel, inputs, stride=1)

  np.testing.assert_allclose(result1, result2)
  print("两种方式结果相等")


if __name__ == "__main__":
  test()
