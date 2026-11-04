package com.vizuara.genai.attention;

import java.util.Arrays;

/**
 * Lectures 07 & 08: Enterprise Java 17+ Bahdanau (Additive) Attention & Seq2Seq Decoder.
 *
 * Maps Neural Attention Mechanics to Distributed Systems Primitives:
 * - Encoder Outputs (H)        : In-Memory Indexed Event Store / Key-Value Table
 * - Decoder Hidden State (s)   : Dynamic Parameterized Search Query
 * - Alignment Energy (e_{t,j}) : Learned Neural Relevance Scorer (L2R)
 * - Softmax Weights (alpha)    : Normalized Scatter-Gather Routing Weights
 * - Context Vector (c_t)       : Dynamic On-Demand Materialized Read Projection
 */
public class BahdanauAttentionSeq2Seq {

    public record AttentionResult(double[] alphaWeights, double[] rawScores, double[] contextVector) {}

    public record DecoderStepResult(double[] logits, double[] nextHidden, double[] nextCell, AttentionResult attention) {}

    public static class BahdanauAttention {
        private final int encHidDim;
        private final int decHidDim;
        private final double[][] W_a; // Shape: decHidDim x (decHidDim + encHidDim)
        private final double[] b_a;   // Shape: decHidDim
        private final double[] v_a;   // Shape: decHidDim (no bias, matching nn.Linear(..., bias=False))

        public BahdanauAttention(int encHidDim, int decHidDim, double[][] W_a, double[] b_a, double[] v_a) {
            this.encHidDim = encHidDim;
            this.decHidDim = decHidDim;
            this.W_a = W_a;
            this.b_a = b_a;
            this.v_a = v_a;
        }

        public AttentionResult forward(double[] decoderHidden, double[][] encoderOutputs) {
            int srcLen = encoderOutputs.length;
            double[] rawScores = new double[srcLen];
            double maxScore = Double.NEGATIVE_INFINITY;

            // 1. Compute additive alignment energy for every source token j in [0 .. srcLen-1]
            for (int j = 0; j < srcLen; j++) {
                double[] h_j = encoderOutputs[j];
                double score = 0.0;

                for (int k = 0; k < decHidDim; k++) {
                    double preAct = b_a[k];
                    // Multiply decoderHidden part
                    for (int d = 0; d < decHidDim; d++) {
                        preAct += W_a[k][d] * decoderHidden[d];
                    }
                    // Multiply encoderOutput part
                    for (int e = 0; e < encHidDim; e++) {
                        preAct += W_a[k][decHidDim + e] * h_j[e];
                    }
                    double energy_k = Math.tanh(preAct);
                    score += v_a[k] * energy_k;
                }
                rawScores[j] = score;
                if (score > maxScore) {
                    maxScore = score;
                }
            }

            // 2. Numerically stable Softmax over source positions
            double[] alpha = new double[srcLen];
            double sumExp = 0.0;
            for (int j = 0; j < srcLen; j++) {
                alpha[j] = Math.exp(rawScores[j] - maxScore);
                sumExp += alpha[j];
            }
            for (int j = 0; j < srcLen; j++) {
                alpha[j] /= sumExp;
            }

            // 3. Batch Matrix Multiply (bmm): context c_t = sum_j alpha_{t,j} * h_j
            double[] context = new double[encHidDim];
            for (int j = 0; j < srcLen; j++) {
                for (int e = 0; e < encHidDim; e++) {
                    context[e] += alpha[j] * encoderOutputs[j][e];
                }
            }

            return new AttentionResult(alpha, rawScores, context);
        }
    }

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" LECTURES 07 & 08: ENTERPRISE JAVA 17+ BAHDANAU ATTENTION ENGINE");
        System.out.println("================================================================================");

        String[] srcTokens = { "<sos>", "i", "am", "student", "<eos>" };

        // 4D Semantic Encoder Outputs: [Subject, Copula-Verb, Noun-Entity, Boundary]
        double[][] encoderOutputs = {
            { 0.10, 0.05, 0.05, 0.85 }, // <sos>
            { 0.95, 0.10, 0.05, 0.05 }, // i
            { 0.10, 0.95, 0.10, 0.05 }, // am
            { 0.05, 0.10, 0.98, 0.05 }, // student
            { 0.05, 0.05, 0.05, 0.95 }  // <eos>
        };

        double[][] W_a = {
            { 2.2, 0.0, 0.0, 0.0,  2.2, 0.0, 0.0, 0.0 },
            { 0.0, 2.2, 0.0, 0.0,  0.0, 2.2, 0.0, 0.0 },
            { 0.0, 0.0, 2.2, 0.0,  0.0, 0.0, 2.2, 0.0 },
            { 0.0, 0.0, 0.0, 2.2,  0.0, 0.0, 0.0, 2.2 }
        };
        double[] b_a = { -2.6, -2.6, -2.6, -2.6 };
        double[] v_a = { 2.5, 2.5, 2.5, 2.5 };

        BahdanauAttention attention = new BahdanauAttention(4, 4, W_a, b_a, v_a);

        String[] frenchTargets = { "je", "suis", "etudiant" };
        double[][] decoderQueries = {
            { 1.0, 0.0, 0.0, 0.0 }, // Querying Subject for "je"
            { 0.0, 1.0, 0.0, 0.0 }, // Querying Verb for "suis"
            { 0.0, 0.0, 1.0, 0.0 }  // Querying Noun for "etudiant"
        };

        for (int t = 0; t < frenchTargets.length; t++) {
            AttentionResult res = attention.forward(decoderQueries[t], encoderOutputs);
            System.out.printf("%nDecoder Step t=%d (Predicting '%s'):%n", (t + 1), frenchTargets[t]);
            for (int j = 0; j < srcTokens.length; j++) {
                System.out.printf("  alpha[%-8s] = %.4f (%.1f%%)%n",
                    srcTokens[j], res.alphaWeights()[j], res.alphaWeights()[j] * 100.0);
            }
            System.out.println("  Dynamic Context c_t = " + Arrays.toString(
                Arrays.stream(res.contextVector()).map(v -> Math.round(v * 10000.0) / 10000.0).toArray()
            ));
        }
    }
}
