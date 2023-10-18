from array import ClArray

import numpy as np
from utils import cl, prod


def elemwise_op(op, **inp):
  x = list(inp.values())[0]
  ret = ClArray(shape=x.shape, dtype=x.dtype)

  args_stride = ", ".join("".join(f"int {x}_s{i}, " for x in inp) + f"int res_s{i}" for i in range(ret.ndim))
  args_offset = ", ".join(f"int {x}_ofst" for x in inp)
  args_input = ", ".join(f"__global const float *inp_{x}" for x in inp)
  args_return = "__global float *ret"
  args = [args_stride, args_offset, args_input, args_return]
  args = ", ".join(a for a in args if a)

  update_expr = "".join(f"idx=ptr/res_s{i}; ptr%=res_s{i}; " + "".join(f"{x}_i+=idx*{x}_s{i}; " for x in inp) for i in range(ret.ndim))
  assign_expr = "".join(f"float {x}=inp_{x}[{x}_i+{x}_ofst]; " for x in inp)
  op = cl.build("ElemwiseOp", f"""__kernel void ElemwiseOp({args}) {{
    {"; ".join(f"int {x}_i=0" for x in inp)};
    int idx=0, gl_id=get_global_id(0); int ptr=gl_id;
    {update_expr}
    {assign_expr}
    ret[gl_id] = {op};
  }}""")
  args_stride = [np.int32(s) for ss in zip(*[x.strides for x in (list(inp.values())+[ret])]) for s in ss]
  args_offset = [np.int32(x.offset) for x in inp.values()]
  args_input = [x.buffer for x in inp.values()]
  args_return = [ret.buffer]
  args = args_stride + args_offset + args_input + args_return
  event = op((prod(x.shape),), None, *args)
  event.wait()
  return ret

def matmul_op(**inp):
  # rule: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
  a, b = inp.values()
  # calculate return shape
  squeezes = []
  if a.ndim == 1:
    a = a.reshape((1, *a.shape))
    squeezes.append(0)
  if b.ndim == 1:
    b = b.reshape((*b.shape, 1))
    squeezes.append(-1)
  ret_shape = tuple((*a.shape[:-1], b.shape[-1]))

  if a.ndim > 3:
    a = a.reshape((prod(a.shape[:-2]), *a.shape[2:]))
  if b.ndim > 3:
    b = b.reshape((prod(b.shape[:-2]), *b.shape[2:]))
  if a.ndim == 2:
    a = a.reshape((1, *a.shape))
  if b.ndim == 2:
    b = b.reshape((1, *b.shape))
  if a.shape[0] != b.shape[0]:
    assert a.shape[0] == 1 or b.shape[0] == 1
    if a.shape[0] == 1 and b.shape[0] != 1:
      a = a.expand((b.shape[0], *a.shape[1:]))
    if b.shape[0] == 1 and a.shape[0] != 1:
      b = b.expand((a.shape[0], *b.shape[1:]))
  assert a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], \
          f"invalid shape for matmul {a.shape} @ {b.shape}"

  ret = ClArray(shape=ret_shape, dtype=a.dtype)
  BS, M, K, N = prod(a.shape[:-2]), a.shape[-2], a.shape[-1], b.shape[-1]
  gs = 1
  while gs <= 8 and M % gs == 0 and N % gs == 0 and K % gs == 0 and gs <= K and gs <= M and gs <= N:
    gs *= 2
  gs //= 2
  op = cl.build("matmul_op", f"""__kernel void matmul_op(int BS, int M, int N, int K,
      {''.join(f'int A_s{i}, int B_s{i}, ' for i in range(3))} int a_ofst, int b_ofst,
      __global const float *A, __global const float *B, __global float *C) {{
    int bs=get_global_id(0), m=get_global_id(1), n=get_global_id(2), i=get_local_id(1), j=get_local_id(2);
    __local float Alcl[{gs}][{gs}], Blcl[{gs}][{gs}];
    float acc = 0.0f;
    for (int t=0; t<K/{gs}; t++) {{
      Alcl[i][j] = A[bs*A_s0+m*A_s1+(t*{gs}+j)*A_s2+a_ofst];
      Blcl[i][j] = B[bs*B_s0+(t*{gs}+i)*B_s1+n*B_s2+b_ofst];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k=0; k<{gs}; k++) {{
        acc += Alcl[i][k] * Blcl[k][j];
      }}
      barrier(CLK_LOCAL_MEM_FENCE);
    }}
    C[bs*M*N+m*N+n] = acc;
  }}""")
  strides = [s for ss in zip(a.strides, b.strides) for s in ss]
  args = [np.int32(x) for x in [BS, M, N, K] + strides + [a.offset, b.offset]]
  event = op((BS, M, N), (1, gs, gs), *args, a.buffer, b.buffer, ret.buffer)
  event.wait()
  for axis in squeezes:
    ret = ret.squeeze(axis)
  return ret

def reduce_op(op, axis=None, keepdims=False, **inp):
  assert op in ("max", "sum"), "Invalid op for reduce. Use `max` or `sum`."
  agg = "A+B" if op == "sum" else "max(A,B)"
  pad = "0.0f" if op == "sum" else "-INFINITY"

  x = list(inp.values())[0]
  x_shp = x.shape
  if axis is None:
    axis, x_shp = 0, (prod(x.shape),)
  size = x_shp[axis]

  grp_size = 2
  max_work_group_size = cl.queue.device.max_work_group_size
  while grp_size != max_work_group_size and grp_size < size:
    grp_size *= 2

  def calculate_ret_shape(x_shp, axis, keepdims, grp_size, n_grps):
    if n_grps <= 1:
      ret_shape = [d for i, d in enumerate(x_shp) if i != axis]
      if keepdims:
        ret_shape.insert(axis, 1)
      return tuple(ret_shape)
    return tuple(n_grps if i == axis else d for i, d in enumerate(x_shp))

  n_grps = (size + grp_size - 1) // grp_size
  ret_shape = calculate_ret_shape(x_shp, axis, keepdims, grp_size, n_grps)
  ret = ClArray(shape=ret_shape, dtype=x.dtype)
  # merge non-target axes
  p1 = [prod(x_shp[:axis])] if axis!=0 else []
  p2 = [prod(x_shp[axis+1:])] if axis!=len(x_shp)-1 else []
  global_size = (*p1, grp_size*n_grps, *p2)
  axis, ndim = len(p1), len(global_size)

  a = [f"gl_id_{i}" for i in range(ndim)]
  b = [f"gl_s_{i}" for i in range(ndim)]
  c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
  gl2lcl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
  a = [(f"grp_id_{i}" if i == axis else f"gl_id_{i}") for i in range(ndim)]
  b = [f"(gl_s_{i}/grp_s_{i})" for i in range(ndim)]
  c = ["*".join(b[i+1:]) for i in range(ndim-1)] + ["1"]
  lcl2gl = "+".join([f"{a_}*{c_}" for a_, c_ in zip(a, c)])
  # NOTE: calculate offset to get the proper global index
  offset = f"gl_id_0*{'0' if axis==0 else '1' if axis==ndim-1 else 'gl_s_2'}*(gl_s_{axis}-size)"
  op = cl.build("reduce_op", f"""__kernel void reduce_op(int size, int ofst, __global const float *inp, __local float *lcl, __global float *ret) {{
    {''.join([f'int gl_id_{i}=get_global_id({i});int gl_s_{i}=get_global_size({i});int grp_id_{i}=get_group_id({i});int grp_s_{i}=get_local_size({i});' for i in range(ndim)])}
    int lcl_id = get_local_id({axis});
    lcl[lcl_id] = gl_id_{axis} < size ? inp[{gl2lcl}-{offset}+ofst] : {pad};
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = grp_s_{axis}>>1; stride > 0; stride>>=1) {{
      float A = lcl[lcl_id], B = lcl[lcl_id+stride];
      if (lcl_id<stride) lcl[lcl_id] = {agg};
      barrier(CLK_LOCAL_MEM_FENCE);
    }}
    if (lcl_id == 0) ret[{lcl2gl}] = lcl[0];
  }}""")
  local_mem = cl.alloc_local(x.dtype().itemsize * grp_size)
  local_size = tuple(grp_size if i == axis else 1 for i in range(ndim))
  event = op(global_size, local_size, np.int32(size), np.int32(x.offset), x.buffer, local_mem, ret.buffer)
  event.wait()
  if n_grps > 1:
    ret = reduce_op(op, axis=axis, keepdims=keepdims, A=ret)
  return ret
