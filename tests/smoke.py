import sys
sys.path.insert(0, "<build>/bindings")
sys.path.insert(0, r"F:/Experiments/Vorticity/MC Fluid Cuda/cmake-build-debug-visual-studio/bindings")
print("sys.path[0] =", sys.path[0])   #
import phash
print("phash smoke-test:", phash.fill_dummy())
