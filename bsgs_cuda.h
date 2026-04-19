#ifndef BSGS_CUDA_H
#define BSGS_CUDA_H

#include "secp256k1/Point.h"
#include "secp256k1/Int.h"
#include <vector>

void run_bsgs_cuda(Int *start, Int *end, std::vector<Point> &targets, int num_targets, bool compressed);

#endif
