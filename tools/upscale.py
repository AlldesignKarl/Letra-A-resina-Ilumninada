"""Upscale source photos x4 with Real-ESRGAN (RRDBNet) via onnxruntime.

Reads the official RealESRGAN_x4plus.pth without PyTorch, builds the
equivalent ONNX graph and runs it tile by tile.
Usage: python3 upscale.py weights.pth in.png out.png
"""
import pickle, sys, zipfile
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from PIL import Image


def load_pth(path):
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0]

    def storage(dtype_name, key):
        dt = {"FloatStorage": np.float32, "HalfStorage": np.float16}[dtype_name]
        return np.frombuffer(zf.read(f"{prefix}/data/{key}"), dtype=dt)

    def rebuild(stor, offset, size, stride, *args):
        if not size:
            return stor[offset:offset + 1].reshape(())
        n = 1 + sum((s - 1) * st for s, st in zip(size, stride))
        flat = stor[offset:offset + n]
        return np.lib.stride_tricks.as_strided(
            flat, shape=size, strides=[st * flat.itemsize for st in stride]).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if name.endswith("Storage"):
                return name
            raise pickle.UnpicklingError(f"{mod}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _loc, _n = pid
            return storage(stype if isinstance(stype, str) else stype, key)

    sd = U(zf.open(f"{prefix}/data.pkl")).load()
    return sd.get("params_ema", sd.get("params", sd))


def build_model(sd):
    nodes, inits = [], []
    cnt = [0]

    def nm(p):
        cnt[0] += 1
        return f"{p}_{cnt[0]}"

    def conv(x, key):
        w, b = key + ".weight", key + ".bias"
        if w not in [i.name for i in inits]:
            inits.append(numpy_helper.from_array(sd[w].astype(np.float32), w))
            inits.append(numpy_helper.from_array(sd[b].astype(np.float32), b))
        y = nm("conv")
        nodes.append(helper.make_node("Conv", [x, w, b], [y], pads=[1, 1, 1, 1]))
        return y

    def lrelu(x):
        y = nm("lr")
        nodes.append(helper.make_node("LeakyRelu", [x], [y], alpha=0.2))
        return y

    def cat(xs):
        y = nm("cat")
        nodes.append(helper.make_node("Concat", xs, [y], axis=1))
        return y

    inits.append(numpy_helper.from_array(np.array(0.2, np.float32), "k02"))

    def res(x, r):
        m = nm("mul")
        nodes.append(helper.make_node("Mul", [r, "k02"], [m]))
        y = nm("add")
        nodes.append(helper.make_node("Add", [m, x], [y]))
        return y

    def add(a, b):
        y = nm("add")
        nodes.append(helper.make_node("Add", [a, b], [y]))
        return y

    inits.append(numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "sc2"))

    def up(x):
        y = nm("up")
        nodes.append(helper.make_node("Resize", [x, "", "sc2"], [y], mode="nearest"))
        return y

    def rdb(x, k):
        x1 = lrelu(conv(x, k + ".conv1"))
        x2 = lrelu(conv(cat([x, x1]), k + ".conv2"))
        x3 = lrelu(conv(cat([x, x1, x2]), k + ".conv3"))
        x4 = lrelu(conv(cat([x, x1, x2, x3]), k + ".conv4"))
        x5 = conv(cat([x, x1, x2, x3, x4]), k + ".conv5")
        return res(x, x5)

    feat = conv("input", "conv_first")
    body = feat
    i = 0
    while f"body.{i}.rdb1.conv1.weight" in sd:
        o = rdb(rdb(rdb(body, f"body.{i}.rdb1"), f"body.{i}.rdb2"), f"body.{i}.rdb3")
        body = res(body, o)
        i += 1
    feat = add(feat, conv(body, "conv_body"))
    feat = lrelu(conv(up(feat), "conv_up1"))
    feat = lrelu(conv(up(feat), "conv_up2"))
    out = conv(lrelu(conv(feat, "conv_hr")), "conv_last")
    nodes.append(helper.make_node("Identity", [out], ["output"]))
    g = helper.make_graph(
        nodes, "rrdb",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, None, None])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, None, None])],
        inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def run(sess, img, tile=192, pad=16, scale=4):
    h, w, _ = img.shape
    out = np.zeros((h * scale, w * scale, 3), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            t = img[py0:py1, px0:px1].transpose(2, 0, 1)[None]
            r = sess.run(None, {"input": np.ascontiguousarray(t)})[0][0].transpose(1, 2, 0)
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                r[oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
            print(f"\r{y0}/{h}", end="", flush=True)
    print()
    return np.clip(out, 0, 1)


if __name__ == "__main__":
    weights, src, dst = sys.argv[1:4]
    model = build_model(load_pth(weights))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    img = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    out = run(sess, img)
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(dst)
