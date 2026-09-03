package com.vizuara.genai.lstm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Lecture 06: Enterprise Java 17+ LSTM Language Model (Next-Word Generator) & Seq2Seq Translator.
 * 
 * Maps Neural Network Text Generation to Distributed Systems Primitives:
 * - Next-Word Generator: Autoregressive Event-Stream Predictor / Search Autocomplete Service
 * - N-Gram Token Sequences: Sliding-Window Audit Log Chunks
 * - Pre-Padding: Fixed-Width Buffer Normalization
 */
public class LSTMLanguageModelAndTranslator {

    public record LSTMContext(double[] hiddenState, double[] cellState) {}

    public static class LSTMLanguageModel {
        private final int vocabSize;
        private final int embedDim;
        private final int hiddenDim;

        private final double[][] embeddings;
        // Gates: Input (i), Forget (f), Candidate (g), Output (o)
        private final double[][] W_i, U_i; private final double[] b_i;
        private final double[][] W_f, U_f; private final double[] b_f;
        private final double[][] W_g, U_g; private final double[] b_g;
        private final double[][] W_o, U_o; private final double[] b_o;
        private final double[][] W_dense; private final double[] b_dense;

        public LSTMLanguageModel(int vocabSize, int embedDim, int hiddenDim) {
            this.vocabSize = vocabSize;
            this.embedDim = embedDim;
            this.hiddenDim = hiddenDim;

            double scale = 1.0 / Math.sqrt(hiddenDim);
            this.embeddings = initMatrix(vocabSize, embedDim, 0.5);

            this.W_i = initMatrix(hiddenDim, embedDim, scale); this.U_i = initMatrix(hiddenDim, hiddenDim, scale); this.b_i = new double[hiddenDim];
            this.W_f = initMatrix(hiddenDim, embedDim, scale); this.U_f = initMatrix(hiddenDim, hiddenDim, scale); this.b_f = initVector(hiddenDim, 1.0);
            this.W_g = initMatrix(hiddenDim, embedDim, scale); this.U_g = initMatrix(hiddenDim, hiddenDim, scale); this.b_g = new double[hiddenDim];
            this.W_o = initMatrix(hiddenDim, embedDim, scale); this.U_o = initMatrix(hiddenDim, hiddenDim, scale); this.b_o = new double[hiddenDim];

            this.W_dense = initMatrix(vocabSize, hiddenDim, scale);
            this.b_dense = new double[vocabSize];
        }

        public record StepOutput(LSTMContext nextContext, double[] logits, int argmaxToken) {}

        public StepOutput step(int tokenId, LSTMContext prevContext) {
            double[] x_t = embeddings[tokenId];
            double[] h_prev = prevContext.hiddenState();
            double[] c_prev = prevContext.cellState();

            double[] i_t = new double[hiddenDim];
            double[] f_t = new double[hiddenDim];
            double[] g_t = new double[hiddenDim];
            double[] o_t = new double[hiddenDim];

            for (int i = 0; i < hiddenDim; i++) {
                double sum_i = b_i[i], sum_f = b_f[i], sum_g = b_g[i], sum_o = b_o[i];
                for (int j = 0; j < embedDim; j++) {
                    sum_i += W_i[i][j] * x_t[j];
                    sum_f += W_f[i][j] * x_t[j];
                    sum_g += W_g[i][j] * x_t[j];
                    sum_o += W_o[i][j] * x_t[j];
                }
                for (int k = 0; k < hiddenDim; k++) {
                    sum_i += U_i[i][k] * h_prev[k];
                    sum_f += U_f[i][k] * h_prev[k];
                    sum_g += U_g[i][k] * h_prev[k];
                    sum_o += U_o[i][k] * h_prev[k];
                }
                i_t[i] = sigmoid(sum_i);
                f_t[i] = sigmoid(sum_f);
                g_t[i] = Math.tanh(sum_g);
                o_t[i] = sigmoid(sum_o);
            }

            // Cell state and hidden state updates
            double[] c_t = new double[hiddenDim];
            double[] h_t = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                c_t[i] = (f_t[i] * c_prev[i]) + (i_t[i] * g_t[i]);
                h_t[i] = o_t[i] * Math.tanh(c_t[i]);
            }

            // Output projection logits
            double[] logits = new double[vocabSize];
            double maxLogit = Double.NEGATIVE_INFINITY;
            int bestToken = 0;
            for (int v = 0; v < vocabSize; v++) {
                double sum = b_dense[v];
                for (int k = 0; k < hiddenDim; k++) sum += W_dense[v][k] * h_t[k];
                logits[v] = sum;
                if (sum > maxLogit) {
                    maxLogit = sum;
                    bestToken = v;
                }
            }

            return new StepOutput(new LSTMContext(h_t, c_t), logits, bestToken);
        }

        public int[] generateNextWords(int[] seedTokens, int numWords) {
            LSTMContext context = new LSTMContext(new double[hiddenDim], new double[hiddenDim]);
            StepOutput output = null;

            // Warm up with seed tokens
            for (int tok : seedTokens) {
                output = step(tok, context);
                context = output.nextContext();
            }

            int[] generated = new int[numWords];
            for (int s = 0; s < numWords; s++) {
                generated[s] = output.argmaxToken();
                output = step(generated[s], context);
                context = output.nextContext();
            }
            return generated;
        }

        private static double sigmoid(double val) {
            val = Math.max(-500.0, Math.min(500.0, val));
            return 1.0 / (1.0 + Math.exp(-val));
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

        private static double[] initVector(int size, double val) {
            double[] v = new double[size];
            Arrays.fill(v, val);
            return v;
        }
    }

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" LECTURE 06: ENTERPRISE JAVA 17+ LSTM TEXT GENERATOR & TRANSLATOR");
        System.out.println("================================================================================");

        String[] vocab = { "<PAD>", "united", "states", "of", "america", "president", "donald", "trump", "new", "york" };
        Map<String, Integer> word2idx = new HashMap<>();
        for (int i = 0; i < vocab.length; i++) word2idx.put(vocab[i], i);

        LSTMLanguageModel model = new LSTMLanguageModel(vocab.length, 4, 8);

        int[] seed = { word2idx.get("united"), word2idx.get("states") };
        int[] generated = model.generateNextWords(seed, 4);

        System.out.print("\nSeed Text: 'united states'\nGenerated Autoregressive Completion: ");
        for (int tok : generated) {
            System.out.print(vocab[tok] + " ");
        }
        System.out.println("\n================================================================================");
    }
}
