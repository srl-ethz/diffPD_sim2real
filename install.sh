# Generate python binding.
cd cpp/core/src
swig -c++ -python py_diff_pd_core.i

# Compile c++ code.
cd ../../
rm -rf build
mkdir build
cd build
cmake ..
make -j4
./diff_pd_demo

# Python binding.
cd ../core/src/
mv py_diff_pd_core.py ../../../python/py_diff_pd/core
mv ../../build/libpy_diff_pd_core.so ../../../python/py_diff_pd/core/_py_diff_pd_core.so