package com.vizuara.genai.assignment02;

import java.util.Arrays;

/**
 * Assignment 02: Enterprise Java 17+ Verification of Handwritten LSTM Next-Word Generator.
 *
 * Settings:
 *   V = 5 ("I", "love", "deep", "learning", "<EOS>")
 *   h = 2 (Hidden & Cell State size)
 *   e = 2 (Token Embedding size)
 *   c = 3 (Context sequence: [0, 1, 2] -> "I love deep")
 */
public class LSTMHandwrittenVerification {

    public record LSTMState(double[] hidden, double[] cell) {}

    private static final String[] VOCAB = { "I", "love", "deep", "learning", "<EOS>" };

    // Embedding Matrix E (5 x 2)
    private static final double[][] E = {
        {  1.0,  0.0 }, // 0: "I"
        {  0.5,  1.0 }, // 1: "love"
        {  0.0,  1.0 }, // 2: "deep"
        { -0.5,  1.0 }, // 3: "learning"
        {  0.0, -1.0 }  // 4: "<EOS>"
    };

    // Concatenated LSTM Gate Weights (2 x 4) & Biases (2 x 1)
    private static final double[][] W_F = { { 0.5,  0.0,  0.5,  0.2 }, {  0.0, 0.5, -0.2, 0.5 } };
    private static final double[]   B_F = { 0.5, 0.5 };

    private static final double[][] W_I = { { 0.4, -0.2,  0.8,  0.2 }, { -0.2, 0.4,  0.2, 0.8 } };
    private static final double[]   B_I = { 0.0, 0.0 };

    private static final double[][] W_O = { { 0.5,  0.2,  0.6,  0.2 }, {  0.2, 0.5,  0.2, 0.6 } };
    private static final double[]   B_O = { 0.1, 0.1 };

    private static final double[][] W_C = { { 0.5,  0.0,  1.0, -0.5 }, {  0.0, 0.5, -0.5, 1.0 } };
    private static final double[]   B_C = { 0.0, 0.0 };

    // Dense Logits Layer W_out (5 x 2) & b_out (5 x 1)
    private static final double[][] W_OUT = {
        {  0.5, -1.0 },
        {  0.8,  0.2 },
        { -0.4,  0.6 },
        { -0.5,  2.0 },
        { -0.2, -0.8 }
    };
    private static final double[] B_OUT = { 0.0, 0.0, 0.1, 0.5, -0.2 };

    private static double sigmoid(double u) {
        return 1.0 / (1.0 + Math.exp(-u));
    }

    private static double[] linearAndActivate(double[][] W, double[] z, double[] b, boolean useSigmoid) {
        double[] out = new double[b.length];
        for (int r = 0; r < b.length; r++) {
            double sum = b[r];
            for (int c = 0; c < z.length; c++) {
                sum += W[r][c] * z[c];
            }
            out[r] = useSigmoid ? sigmoid(sum) : Math.tanh(sum);
        }
        return out;
    }

    public static LSTMState step(double[] x_t, LSTMState prev) {
        // Concatenate [h_{t-1}, x_t] -> shape (4 x 1)
        double[] z_t = new double[]{
            prev.hidden()[0], prev.hidden()[1],
            x_t[0], x_t[1]
        };

        double[] f_t = linearAndActivate(W_F, z_t, B_F, true);
        double[] i_t = linearAndActivate(W_I, z_t, B_I, true);
        double[] o_t = linearAndActivate(W_O, z_t, B_O, true);
        double[] cTilde = linearAndActivate(W_C, z_t, B_C, false);

        // Additive Write-Ahead Log (WAL) Cell State Update
        double[] C_t = new double[2];
        double[] h_t = new double[2];
        for (int k = 0; k < 2; k++) {
            C_t[k] = (f_t[k] * prev.cell()[k]) + (i_t[k] * cTilde[k]);
            h_t[k] = o_t[k] * Math.tanh(C_t[k]);
        }
        return new LSTMState(h_t, C_t);
    }

    public static void main(String[] args) {
        int[] contextIds = { 0, 1, 2 }; // "I", "love", "deep"
        LSTMState state = new LSTMState(new double[]{ 0.0, 0.0 }, new double[]{ 0.0, 0.0 });

        System.out.println("============================================================================");
        System.out.println(" ASSIGNMENT 02: JAVA 17+ VERIFICATION OF HANDWRITTEN LSTM NEXT-WORD MODEL");
        System.out.println("============================================================================");

        for (int t = 0; t < contextIds.length; t++) {
            int tokId = contextIds[t];
            state = step(E[tokId], state);
            System.out.printf("Step t=%d ('%-4s') -> C_%d=%s, h_%d=%s%n",
                (t + 1), VOCAB[tokId],
                (t + 1), fmt(state.cell()),
                (t + 1), fmt(state.hidden()));
        }

        // Dense Logits y = W_out * h_3 + b_out
        double[] y = new double[VOCAB.length];
        double sumExp = 0.0;
        double[] expY = new double[VOCAB.length];
        for (int k = 0; k < VOCAB.length; k++) {
            y[k] = W_OUT[k][0] * state.hidden()[0] + W_OUT[k][1] * state.hidden()[1] + B_OUT[k];
            expY[k] = Math.exp(y[k]);
            sumExp += expY[k];
        }

        int bestIdx = 0;
        double bestProb = -1.0;
        System.out.println("\n--- Final Logits & Softmax Distribution ---");
        for (int k = 0; k < VOCAB.length; k++) {
            double prob = expY[k] / sumExp;
            if (prob > bestProb) {
                bestProb = prob;
                bestIdx = k;
            }
            System.out.printf("  ID %d ('%-8s'): logit = %7.4f | prob = %.4f (%.2f%%)%n",
                k, VOCAB[k], y[k], prob, prob * 100.0);
        }
        System.out.printf("%nPredicted Next Word: ID %d ('%s')%n", bestIdx, VOCAB[bestIdx]);
    }

    private static String fmt(double[] v) {
        return String.format("[%.4f, %.4f]", v[0], v[1]);
    }
}
