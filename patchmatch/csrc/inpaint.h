#pragma once

#include <vector>

#include "masked_image.h"
#include "nnf.h"

class Inpainting {
public:
    Inpainting(mmwrap_Matrix image, mmwrap_Matrix mask, const PatchDistanceMetric *metric);
    Inpainting(mmwrap_Matrix image, mmwrap_Matrix mask, mmwrap_Matrix global_mask, const PatchDistanceMetric *metric);
    mmwrap_Matrix run(bool verbose = false, bool verbose_visualize = false, unsigned int random_seed = 1212);

private:
    void _initialize_pyramid(void);
    MaskedImage _expectation_maximization(MaskedImage source, MaskedImage target, int level, bool verbose);
    void _expectation_step(const NearestNeighborField &nnf, bool source2target, mmwrap_Matrix &vote, const MaskedImage &source, bool upscaled);
    void _maximization_step(MaskedImage &target, const mmwrap_Matrix &vote);

    MaskedImage m_initial;
    std::vector<MaskedImage> m_pyramid;

    NearestNeighborField m_source2target;
    NearestNeighborField m_target2source;
    const PatchDistanceMetric *m_distance_metric;
};

