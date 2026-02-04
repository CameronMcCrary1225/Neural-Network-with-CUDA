#pragma once

#include "base.h"           // f32, u64 types
#include "MachineLearning.h"// matrix type (or include the header where matrix is declared)
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

	// Host-callable wrappers (call these from CPU code)
	void cuda_hello();
	void mat_add_cuda(matrix* out, const matrix* a, const matrix* b);
	void mat_sub_cuda(matrix* out, const matrix* a, const matrix* b);
	void mat_relu_cuda(matrix* out, const matrix* in);
	void mat_softmax_cuda(matrix* out, const matrix* in);
	void mat_cross_entropy_cuda(matrix* out, const matrix* p, const matrix* q);
	void mat_mul_cuda(matrix* out, const matrix* a, const matrix* b);
#ifdef __cplusplus
}
#endif