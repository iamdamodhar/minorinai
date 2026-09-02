package com.vizuara.genai.rnn;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Module 01: Vanilla RNN from Scratch in Enterprise Java 17+.
 * Demonstrates:
 * 1. Step-by-step unrolling of a Recurrent Hidden State across a token sequence.
 * 2. Exact matrix computations: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h).
 * 3. Softmax next-token probability distribution: y_t = softmax(W_hy @ h_t + b_y).
 * 4. Mapping to Backend Distributed Systems mutable session state cache.
 */
public class RNNFromScratch {

    public record RNNStepResult(double[] hiddenState, double[] probabilities) {}

    public static class VanillaRNNCell {
        private final int inputDim;
        private final int hiddenDim;
        private final int vocabSize;

        private final double[][] W_xh;
        private final double[][] W_hh;
        private final double[] b_h;

        private final double[][] W_hy;
        private final double[] b_y;

        public VanillaRNNCell(int inputDim, int hiddenDim, int vocabSize) {
            this.inputDim = inputDim;
            this.hiddenDim = hiddenDim;
            this.vocabSize = vocabSize;

            double scale = 1.0 / Math.sqrt(hiddenDim);
            this.W_xh = initMatrix(hiddenDim, inputDim, scale);
            this.W_hh = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_h = new double[hiddenDim];

            this.W_hy = initMatrix(vocabSize, hiddenDim, scale);
            this.b_y = new double[vocabSize];
        }

        public RNNStepResult step(double[] x_t, double[] h_prev) {
            // 1. Linear combination: W_xh @ x_t + W_hh @ h_{t-1} + b_h
            double[] h_t = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                double sum = b_h[i];
                for (int j = 0; j < inputDim; j++) sum += W_xh[i][j] * x_t[j];
                for (int k = 0; k < hiddenDim; k++) sum += W_hh[i][k] * h_prev[k];
                h_t[i] = Math.tanh(sum);
            }

            // 2. Output projection logits: W_hy @ h_t + b_y
            double[] logits = new double[vocabSize];
            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < vocabSize; i++) {
                double sum = b_y[i];
                for (int k = 0; k < hiddenDim; k++) sum += W_hy[i][k] * h_t[k];
                logits[i] = sum;
                if (sum > maxLogit) maxLogit = sum;
            }

            // 3. Numerically stable Softmax
            double sumExp = 0.0;
            double[] probs = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                probs[i] = Math.exp(logits[i] - maxLogit);
                sumExp += probs[i];
            }
            for (int i = 0; i < vocabSize; i++) {
                probs[i] /= sumExp;
            }

            return new RNNStepResult(h_t, probs);
        }

        private static double[][] initMatrix(int rows, int cols, double scale) {
            double[][] m = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    m[i][j] = (Math.random() * 2 - 1) * scale;
                }
            }
            return m;
        }
    }

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" MODULE 01: ENTERPRISE JAVA 17+ VANILLA RNN FORWARD PASS");
        System.out.println("================================================================================");

        int inputDim = 4;
        int hiddenDim = 5;
        int vocabSize = 6;

        VanillaRNNCell rnn = new VanillaRNNCell(inputDim, hiddenDim, vocabSize);

        // Retail Category Clickstream: [Formality, Warmth, PriceTier, Category]
        Map<String, double[]> catalog = new HashMap<>();
        catalog.put("The",       new double[]{ 0.1,  0.0,  0.0,  0.0 });
        catalog.put("chef",      new double[]{ 0.8,  0.2,  0.5,  0.1 });
        catalog.put("cooked",    new double[]{ 0.0,  0.9, -0.2,  0.0 });
        catalog.put("the",       new double[]{ 0.1,  0.0,  0.0,  0.0 });
        catalog.put("delicious", new double[]{ 0.3,  0.1,  0.9,  0.4 });

        String[] sentence = { "The", "chef", "cooked", "the", "delicious" };
        double[] h_curr = new double[hiddenDim]; // h_0 = zeros

        System.out.println("\nUnrolling RNN over sequence: " + String.join(" -> ", sentence) + "\n");

        for (int t = 0; t < sentence.length; t++) {
            String token = sentence[t];
            double[] x_t = catalog.get(token);

            RNNStepResult result = rnn.step(x_t, h_curr);
            h_curr = result.hiddenState();

            double norm = 0.0;
            for (double v : h_curr) norm += v * v;
            norm = Math.sqrt(norm);

            System.out.printf("Time Step t=%d | Token: '%-9s' | Hidden State h_%d norm: %.4f | Vector: %s%n",
                (t + 1), token, (t + 1), norm, formatVec(h_curr));
        }

        System.out.println("\n================================================================================");
        System.out.println(" THE PRESSURE ON THE LAST HIDDEN LAYER (h_T / Information Bottleneck)");
        System.out.println("================================================================================");
        System.out.println("Final summary vector h_5 (size: " + hiddenDim + "): " + formatVec(h_curr));
        System.out.println("Backend Analogy: Compressing the entire session history into a single mutable");
        System.out.println("in-memory cache causes catastrophic forgetting of early events ('The chef').");
        System.out.println("================================================================================");
    }

    private static String formatVec(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(String.format("%.3f", v[i]));
            if (i < v.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
