#include "models.h"

ggml_cgraph * clip_graph_ernie45vlmoe::build() {
    // ERNIE-4.5-VL-MoE Vision + Resampler:
    // 1. ViT encoder with 2D position embeddings and M-RoPE support
    // 2. Resampler with spatial conv (2x2 grouping) + optional temporal + MLP + RMS norm

    const int batch_size = 1;
    const int n_pos = n_patches;
    const int spatial_conv_size = hparams.spatial_conv_size;   // 2
    const int temporal_conv_size = hparams.temporal_conv_size; // 2
    const bool use_temporal = hparams.use_temporal_conv;

    GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");
    GGML_ASSERT(spatial_conv_size == 2 && "ERNIE-4.5-VL-MoE requires spatial_conv_size=2");

    // Build vision encoder with patch embedding
    // Note: patch_embeddings_0 is reshaped to 4D during export for conv2d compatibility
    ggml_tensor * inp = build_inp();

    // ERNIE-4.5-VL uses RoPE (Rotary Position Embedding), not learned position embeddings
    // Position encoding is applied within attention layers via RoPE
    // So we don't need to add position embeddings here

    ggml_tensor * inpL = inp;

    // Pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // Loop over encoder layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        // Layernorm 1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ln1", il);

        // Self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // Residual
        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        // Layernorm 2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // FFN
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // Residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // Post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    ggml_tensor * embeddings = inpL;
    cb(embeddings, "vision_output", -1);

    // -------------------------------------------
    // Resampler projection
    // -------------------------------------------
    // Input shape: [n_embd, n_patches] = [1280, 1600]
    // We need to group 2x2 patches: 40x40 patches -> 20x20 groups
    // Output shape: [n_embd*4, n_groups] = [5120, 400]

    const int n_groups_x = n_patches_x / spatial_conv_size;  // 40/2 = 20
    const int n_groups_y = n_patches_y / spatial_conv_size;  // 40/2 = 20
    const int n_groups   = n_groups_x * n_groups_y;          // 400

    // Use patch_merge_permute to group 2x2 patches
    // Note: build_patch_merge_permute expects 2D input [n_embd, n_patches]
    embeddings = build_patch_merge_permute(embeddings, spatial_conv_size);

    // embeddings is now [n_embd*4, n_groups] = [5120, 400]
    cb(embeddings, "spatial_reshape", -1);

    // Spatial linear path: Linear -> GELU -> Linear -> LayerNorm
    ggml_tensor * spatial_out = embeddings;

    // First linear: [5120, 5120] x [5120, n_groups] -> [5120, n_groups]
    spatial_out = ggml_mul_mat(ctx0, model.mm_spatial_0_w, spatial_out);
    spatial_out = ggml_add(ctx0, spatial_out, model.mm_spatial_0_b);
    cb(spatial_out, "spatial_linear_0", -1);

    // GELU
    spatial_out = ggml_gelu(ctx0, spatial_out);
    cb(spatial_out, "spatial_gelu", -1);

    // Second linear
    spatial_out = ggml_mul_mat(ctx0, model.mm_spatial_2_w, spatial_out);
    spatial_out = ggml_add(ctx0, spatial_out, model.mm_spatial_2_b);
    cb(spatial_out, "spatial_linear_2", -1);

    // LayerNorm
    spatial_out = build_norm(spatial_out, model.mm_spatial_norm_w, model.mm_spatial_norm_b, NORM_TYPE_NORMAL, eps, -1);
    cb(spatial_out, "spatial_norm", -1);

    ggml_tensor * resampler_out = spatial_out;

    // Debug: check shapes before MLP
    GGML_ASSERT(resampler_out->ne[0] == n_embd * spatial_conv_size * spatial_conv_size); // should be 5120
    GGML_ASSERT(resampler_out->ne[1] == n_groups); // should be 400

    // Temporal path (skip for single images)
    if (use_temporal && model.mm_temp_0_w) {
        // For video: concatenate adjacent temporal frames
        // Since we're processing single images, this path is unused.
        // Implementation would require grid_thw and complex indexing.
        LOG_WRN("%s: temporal convolution enabled but not implemented for single images\n", __func__);
    }

    // Final MLP: Linear + RMSNorm
    resampler_out = ggml_mul_mat(ctx0, model.mm_mlp_w, resampler_out);
    resampler_out = ggml_add(ctx0, resampler_out, model.mm_mlp_b);
    cb(resampler_out, "mlp", -1);

    // RMS norm (final output normalization)
    resampler_out = build_norm(resampler_out, model.mm_after_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cb(resampler_out, "after_norm", -1);

    // Build the graph
    ggml_build_forward_expand(gf, resampler_out);

    return gf;
}
