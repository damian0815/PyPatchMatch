#pragma once

#include "matrixmath_wrapper.h"

class MaskedImage {
public:
    MaskedImage() : m_image(), m_mask(), m_global_mask(), m_image_grady(), m_image_gradx(), m_image_grad_computed(false) {
        // pass
    }
    MaskedImage(mmwrap_Matrix image, mmwrap_Matrix mask) : m_image(image), m_mask(mask), m_image_grad_computed(false) {
        // pass
    }
    MaskedImage(mmwrap_Matrix image, mmwrap_Matrix mask, mmwrap_Matrix global_mask) : m_image(image), m_mask(mask), m_global_mask(global_mask), m_image_grad_computed(false) {
        // pass
    }
    MaskedImage(mmwrap_Matrix image, mmwrap_Matrix mask, mmwrap_Matrix global_mask, mmwrap_Matrix grady, mmwrap_Matrix gradx, bool grad_computed) :
        m_image(image), m_mask(mask), m_global_mask(global_mask),
        m_image_grady(grady), m_image_gradx(gradx), m_image_grad_computed(grad_computed) {
        // pass
    }
    MaskedImage(int width, int height) : m_global_mask(), m_image_grady(), m_image_gradx() {
        m_image = mmwrap_Matrix(mmwrap_Size(width, height), CV_8UC3);
        mmwrap_clear(m_image);

        m_mask = mmwrap_Matrix(mmwrap_Size(width, height), CV_8U);
        mmwrap_clear(m_image);
    }
    inline MaskedImage clone() {
        return MaskedImage(
            m_image.clone(), m_mask.clone(), m_global_mask.clone(),
            m_image_grady.clone(), m_image_gradx.clone(), m_image_grad_computed
        );
    }

    inline mmwrap_Size size() const {
        return m_image.size();
    }
    inline const mmwrap_Matrix &image() const {
        return m_image;
    }
    inline const mmwrap_Matrix &mask() const {
        return m_mask;
    }
    inline const mmwrap_Matrix &global_mask() const {
        return m_global_mask;
    }
    inline const mmwrap_Matrix &grady() const {
        assert(m_image_grad_computed);
        return m_image_grady;
    }
    inline const mmwrap_Matrix &gradx() const {
        assert(m_image_grad_computed);
        return m_image_gradx;
    }

    inline void init_global_mask_mat() {
        m_global_mask = mmwrap_Matrix(m_mask.size(), CV_8U);
        mmwrap_clear(m_global_mask);
    }
    inline void set_global_mask_mat(const mmwrap_Matrix &other) {
        m_global_mask = other;
    }

    inline bool is_masked(int y, int x) const {
        return static_cast<bool>(m_mask.at<unsigned char>(y, x));
    }
    inline bool is_globally_masked(int y, int x) const {
        return !m_global_mask.empty() && static_cast<bool>(m_global_mask.at<unsigned char>(y, x));
    }
    inline void set_mask(int y, int x, bool value) {
        m_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }
    inline void set_global_mask(int y, int x, bool value) {
        m_global_mask.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
    }
    inline void clear_mask() {
        mmwrap_clear(m_mask);
    }

    inline const unsigned char *get_image(int y, int x) const {
        return m_image.ptr<unsigned char>(y, x);
    }
    inline unsigned char *get_mutable_image(int y, int x) {
        return m_image.ptr<unsigned char>(y, x);
    }

    inline unsigned char get_image(int y, int x, int c) const {
        return m_image.ptr<unsigned char>(y, x)[c];
    }
    inline int get_image_int(int y, int x, int c) const {
        return static_cast<int>(m_image.ptr<unsigned char>(y, x)[c]);
    }

    bool contains_mask(int y, int x, int patch_size) const;
    MaskedImage downsample() const;
    MaskedImage upsample(int new_w, int new_h) const;
    MaskedImage upsample(int new_w, int new_h, const mmwrap_Matrix &new_global_mask) const;
    void compute_image_gradients();
    void compute_image_gradients() const;

    static const mmwrap_Size kDownsampleKernelSize;
    static const int kDownsampleKernel[6];

private:
	mmwrap_Matrix m_image;
	mmwrap_Matrix m_mask;
    mmwrap_Matrix m_global_mask;
    mmwrap_Matrix m_image_grady;
    mmwrap_Matrix m_image_gradx;
    bool m_image_grad_computed = false;
};

