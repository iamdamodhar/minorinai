package com.vizuara.genai.transformers;

import java.util.Arrays;

/**
 * Transformers Architecture: 3-Phase Pipeline in Enterprise Java 17+.
 *
 * Traces a token sequence through:
 * - Phase 1 (InputPipeline)    : Tokenized Text -> Token Embedding (E_tok) + Positional Embedding (E_pos)
 * - Phase 2 (ProcessorBlock)   : LayerNorm 1 -> Multi-Head Causal Self-Attention -> Dropout -> Residual 1
 *                                -> LayerNorm 2 -> Feed-Forward Network (GELU) -> Dropout -> Residual 2
 * - Phase 3 (OutputHead)       : Final LayerNorm -> Linear Output Projection -> Temperature Softmax
 */
public class TransformerThreePhases {

    public record InputPhaseResult(int[] tokenIds, double[][] tokenEmbeddings, double[][] posEmbeddings, double[][] combinedInput) {}

    public record ProcessorPhaseResult(
        double[][] ln1Out,
        double[][][] headAttentionWeights,
        double[][] mhaOut,
        double[][] residual1,
        double[][] ln2Out,
        double[][] ffnOut,
        double[][] residual2
    ) {}

    public record OutputPhaseResult(double[][] finalLayerNormOut, double[][] logits, double[] nextTokenProbs, int predictedTokenId) {}

    // =========================================================================
    // LINEAR ALGEBRA & ACTIVATION PRIMITIVES
    // =========================================================================
    public static double[][] matmul(double[][] A, double[][] B) {
        int m = A.length;
        int k = A[0].length;
        int n = B[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] += A[i][p] * B[p][j];
                }
            }
        }
        return C;
    }

    public static double[][] transpose(double[][] M) {
        int r = M.length;
        int c = M[0].length;
        double[][] T = new double[c][r];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                T[j][i] = M[i][j];
            }
        }
        return T;
    }

    public static double[][] addMatrices(double[][] A, double[][] B) {
        int r = A.length;
        int c = A[0].length;
        double[][] out = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                out[i][j] = A[i][j] + B[i][j];
            }
        }
        return out;
    }

    public static double[][] layerNorm(double[][] X, double[] gamma, double[] beta, double eps) {
        int seqLen = X.length;
        int dModel = X[0].length;
        double[][] out = new double[seqLen][dModel];

        for (int t = 0; t < seqLen; t++) {
            double mean = 0.0;
            for (int d = 0; d < dModel; d++) {
                mean += X[t][d];
            }
            mean /= dModel;

            double var = 0.0;
            for (int d = 0; d < dModel; d++) {
                double diff = X[t][d] - mean;
                var += diff * diff;
            }
            var /= dModel;
            double invStd = 1.0 / Math.sqrt(var + eps);

            for (int d = 0; d < dModel; d++) {
                out[t][d] = gamma[d] * ((X[t][d] - mean) * invStd) + beta[d];
            }
        }
        return out;
    }

    public static double gelu(double x) {
        // High-precision tanh approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double c = Math.sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }

    public static double[] softmax(double[] scores, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (double s : scores) {
            double scaled = s / temperature;
            if (scaled > -1e8 && scaled > max) {
                max = scaled;
            }
        }
        double[] probs = new double[scores.length];
        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            double scaled = scores[i] / temperature;
            probs[i] = (scaled <= -1e8) ? 0.0 : Math.exp(scaled - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    // =========================================================================
    // PHASE 1: INPUT PIPELINE (Token Embedding + Sinusoidal Positional Encoding)
    // =========================================================================
    public static InputPhaseResult runPhase1Input(int[] tokenIds, double[][] embeddingTable) {
        int seqLen = tokenIds.length;
        int dModel = embeddingTable[0].length;
        double[][] eTok = new double[seqLen][dModel];
        double[][] ePos = new double[seqLen][dModel];

        for (int pos = 0; pos < seqLen; pos++) {
            System.arraycopy(embeddingTable[tokenIds[pos]], 0, eTok[pos], 0, dModel);
            for (int i = 0; i < dModel; i += 2) {
                double divTerm = Math.pow(10000.0, (double) i / dModel);
                ePos[pos][i] = Math.sin(pos / divTerm);
                if (i + 1 < dModel) {
                    ePos[pos][i + 1] = Math.cos(pos / divTerm);
                }
            }
        }
        return new InputPhaseResult(tokenIds, eTok, ePos, addMatrices(eTok, ePos));
    }

    // =========================================================================
    // PHASE 2: PROCESSOR BLOCK (LN1 -> MHA -> Dropout -> Res1 -> LN2 -> FFN -> Res2)
    // =========================================================================
    public static ProcessorPhaseResult runPhase2Processor(
        double[][] xIn,
        double[][][] wQHeads,
        double[][][] wKHeads,
        double[][][] wVHeads,
        double[][] wO,
        double[][] wFf1,
        double[] bFf1,
        double[][] wFf2,
        double[] bFf2
    ) {
        int seqLen = xIn.length;
        int dModel = xIn[0].length;
        int numHeads = wQHeads.length;
        int dK = dModel / numHeads;

        double[] gamma = new double[dModel];
        double[] beta = new double[dModel];
        Arrays.fill(gamma, 1.0);

        // 1. LayerNorm 1
        double[][] z1 = layerNorm(xIn, gamma, beta, 1e-5);

        // 2. Multi-Head Causal Self-Attention
        double[][][] headWeights = new double[numHeads][seqLen][seqLen];
        double[][] concatHeads = new double[seqLen][dModel];

        for (int h = 0; h < numHeads; h++) {
            double[][] Q = matmul(z1, wQHeads[h]);
            double[][] K = matmul(z1, wKHeads[h]);
            double[][] V = matmul(z1, wVHeads[h]);
            double[][] scores = matmul(Q, transpose(K));

            for (int i = 0; i < seqLen; i++) {
                double[] maskedRow = new double[seqLen];
                for (int j = 0; j < seqLen; j++) {
                    maskedRow[j] = (j <= i) ? (scores[i][j] / Math.sqrt(dK)) : -1e9;
                }
                headWeights[h][i] = softmax(maskedRow, 1.0);
            }
            double[][] headOut = matmul(headWeights[h], V);
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(headOut[t], 0, concatHeads[t], h * dK, dK);
            }
        }

        double[][] mhaOut = matmul(concatHeads, wO);

        // 3. Residual Connection 1 (Inference Dropout = Identity)
        double[][] x1 = addMatrices(xIn, mhaOut);

        // 4. LayerNorm 2
        double[][] z2 = layerNorm(x1, gamma, beta, 1e-5);

        // 5. Position-Wise Feed-Forward Network (GELU)
        double[][] ffnHidden = matmul(z2, wFf1);
        for (int t = 0; t < seqLen; t++) {
            for (int c = 0; c < ffnHidden[0].length; c++) {
                ffnHidden[t][c] = gelu(ffnHidden[t][c] + bFf1[c]);
            }
        }
        double[][] ffnOut = matmul(ffnHidden, wFf2);
        for (int t = 0; t < seqLen; t++) {
            for (int c = 0; c < dModel; c++) {
                ffnOut[t][c] += bFf2[c];
            }
        }

        // 6. Residual Connection 2
        double[][] x2 = addMatrices(x1, ffnOut);

        return new ProcessorPhaseResult(z1, headWeights, mhaOut, x1, z2, ffnOut, x2);
    }

    // =========================================================================
    // PHASE 3: OUTPUT HEAD (Final LayerNorm -> Output Linear Layer -> Softmax)
    // =========================================================================
    public static OutputPhaseResult runPhase3Output(double[][] xProcessorOut, double[][] outputWeights, double[] outputBias) {
        int dModel = xProcessorOut[0].length;
        double[] gamma = new double[dModel];
        double[] beta = new double[dModel];
        Arrays.fill(gamma, 1.0);

        double[][] hFinal = layerNorm(xProcessorOut, gamma, beta, 1e-5);
        double[][] logits = matmul(hFinal, outputWeights);

        for (int t = 0; t < logits.length; t++) {
            for (int v = 0; v < outputBias.length; v++) {
                logits[t][v] += outputBias[v];
            }
        }

        double[] nextTokenProbs = softmax(logits[logits.length - 1], 1.0);
        int bestId = 0;
        for (int v = 1; v < nextTokenProbs.length; v++) {
            if (nextTokenProbs[v] > nextTokenProbs[bestId]) {
                bestId = v;
            }
        }
        return new OutputPhaseResult(hFinal, logits, nextTokenProbs, bestId);
    }
}
