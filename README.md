# staple
c++ implementation of tracking algorithm staple: described in the CVPR16 paper "Staple: Complementary Learners for Real-Time Tracking" by Bertinetto et al.

original project:
  https://github.com/xuduo35/STAPLE

some code from:
  https://github.com/foolwood/KCF
 
# Performance optimization and modification:
1. Use fftw and egein lib instead opencv's dft and mat ops in trans filter.
2. Resize big image patch to smaller one in scale filter to reduce compute magnititude of scale filter.
3. Descrease frequency of color histogram model.
4. Adjust original roi's width or height to make it is a sqare region.
5. Add linux cmake configuration.

# Bug fix
1. Over-bound memory access of simd instruction.
2. Compilation error in gcc.
3. Runtime segment fault because of compiler optimization for branch predict.

# Bench Mark
|       Enviroment       | FPS
| :--------------------: | :--:
| 2.8 GHz Intel Core i5 | 450 
| Quad-core Cortex-A53 | 75
