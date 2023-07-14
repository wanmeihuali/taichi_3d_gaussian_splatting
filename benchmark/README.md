# current profiling findings
I've try to adapt the official implementation's output for my renderer, although their color is not activated by sigmoid, it shall not affect the performance on speed. All tests are done on my implementation with 3090.
- Official implementation result 7000 iterations: 1027875 points, inference time: 18.8255625 ms
    - 64.9% rasterization 9.193 ms
    - 11.8% radix sort downsweep 
    - 6.9% generate_point_sort_key 978 us
    - 6.1% generate_num_overlap_tiles 861 us
- Official implementation result 30000 iterations: 2077829 points, inference time: 22.035802734375 ms
    - 49.9% rasterization 8.602 ms
    - 14.9% radix sort downsweep 
    - 11.6% generate_point_sort_key 2.034 ms
    - 10.8% generate_num_overlap_tiles 1.882 ms
- My implementation 30000 iterations: 4.6e5 points, inference time: 20.073513671875 ms
    - 49.8% rasterization 7.554 ms
    - 16.5% generate_point_sort_key 2.499 ms
    - 15.1% generate_num_overlap_tiles 2.284 ms
    - 8.0% radix sort downsweep 

From the observations, my thoughts are:
- many of the points in OFFICIAL implementation are not only useless to PSNR, but also directly skipped during inference. So having such points does not affect the speed.
- If the points skipped does not affect the speed, it is possible to simply the generate_num_overlap_tiles and generate_sort_key_by_num_overlap_tiles functions, of course it will cause usage of more temporary memory. Right now the two functions takes about 30% of the inference time.

Further optimization:
- My implementation with simple gen_tile function 30000 iterations: 4.6e5 points, inference time: 15.84294140625 ms
    - 66.5% rasterization 7.705 ms
    - 14.7% radix sort downsweep 
    - 4.8% elementwise_kernel
    - 3.8% radix sort upsweep
    - 2.9% generate_point_sort_key 2.499 ms

- Official implementation result with simple gen_tile function 7000 iterations: 1027875 points, inference time: 13.411537109375 ms
- Official implementation result with simple gen_tile function 30000 iterations: 2077829 points, inference time: 15.0147490234375 ms
- My implementation with simple gen_tile function and syncthreads_all 30000 iterations: 4.6e5 points, inference time: 16.3198212890625 ms
- Official implementation result with simple gen_tile function and syncthreads_all 7000 iterations: 1027875 points, inference time: 13.135244140625 ms

