package com.vizuara.genai.lstm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Lecture 06: Dedicated LSTM Seq2Seq Language Translator in Enterprise Java 17+.
 * 
 * Demonstrates:
 * 1. Dual-Context Vector State hand-off: (h_enc, c_enc) -> Decoder.
 * 2. 4 Gating equations: f_t (Forget), i_t (Input), g_t (Candidate), o_t (Output).
 * 3. Teacher Forcing training representation and Autoregressive inference generation.
 */
public class LSTMLanguageTranslator {

    public record DualLSTMState(double[] hiddenState, double[] cellState) {}

    public static class TinyEncoderLSTM {
        private final int hiddenSize;
        private final int embedSize;
        private final double[][] embedding;

        // Gates
        private final double[][] W_i, U_i; private final double[] b_i;
        private final double[][] W_f, U_f; private final double[] b_f;
        private final double[][] W_o, U_o; private final double[] b_o;
        private final double[][] W_g, U_g; private final double[] b_g;

        public TinyEncoderLSTM(int vocabSize, int embedSize, int hiddenSize) {
            this.hiddenSize = hiddenSize;
            this.embedSize = embedSize;

            double scale = 1.0 / Math.sqrt(hiddenSize);
            this.embedding = initMatrix(vocabSize, embedSize, 0.5);

            this.W_i = initMatrix(hiddenSize, embedSize, scale); this.U_i = initMatrix(hiddenSize, hiddenSize, scale); this.b_i = new double[hiddenSize];
            this.W_f = initMatrix(hiddenSize, embedSize, scale); this.U_f = initMatrix(hiddenSize, hiddenSize, scale); this.b_f = initVector(hiddenSize, 1.0);
            this.W_o = initMatrix(hiddenSize, embedSize, scale); this.U_o = initMatrix(hiddenSize, hiddenSize, scale); this.b_o = new double[hiddenSize];
            this.W_g = initMatrix(hiddenSize, embedSize, scale); this.U_g = initMatrix(hiddenSize, hiddenSize, scale); this.b_g = new double[hiddenSize];
        }

        public DualLSTMState encode(int[] srcTokens) {
            double[] h = new double[hiddenSize];
            double[] c = new double[hiddenSize];

            for (int token : srcTokens) {
                double[] x_t = embedding[token];
                double[] i_t = new double[hiddenSize];
                double[] f_t = new double[hiddenSize];
                double[] o_t = new double[hiddenSize];
                double[] g_t = new double[hiddenSize];

                for (int i = 0; i < hiddenSize; i++) {
                    double sum_i = b_i[i], sum_f = b_f[i], sum_o = b_o[i], sum_g = b_g[i];
                    for (int j = 0; j < embedSize; j++) {
                        sum_i += W_i[i][j] * x_t[j];
                        sum_f += W_f[i][j] * x_t[j];
                        sum_o += W_o[i][j] * x_t[j];
                        sum_g += W_g[i][j] * x_t[j];
                    }
                    for (int k = 0; k < hiddenSize; k++) {
                        sum_i += U_i[i][k] * h[k];
                        sum_f += U_f[i][k] * h[k];
                        sum_o += U_o[i][k] * h[k];
                        sum_g += U_g[i][k] * h[k];
                    }
                    i_t[i] = sigmoid(sum_i);
                    f_t[i] = sigmoid(sum_f);
                    o_t[i] = sigmoid(sum_o);
                    g_t[i] = Math.tanh(sum_g);
                }

                // Additive Write-Ahead Log (WAL) update
                double[] next_c = new double[hiddenSize];
                double[] next_h = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    next_c[i] = (f_t[i] * c[i]) + (i_t[i] * g_t[i]);
                    next_h[i] = o_t[i] * Math.tanh(next_c[i]);
                }
                c = next_c;
                h = next_h;
            }
            return new DualLSTMState(h, c);
        }
    }

    public static class TinyDecoderLSTM {
        private final int hiddenSize;
        private final int embedSize;
        private final int vocabSize;
        private final double[][] embedding;

        private final double[][] W_i, U_i; private final double[] b_i;
        private final double[][] W_f, U_f; private final double[] b_f;
        private final double[][] W_o, U_o; private final double[] b_o;
        private final double[][] W_g, U_g; private final double[] b_g;

        private final double[][] W_out;
        private final double[] b_out;

        public TinyDecoderLSTM(int vocabSize, int embedSize, int hiddenSize) {
            this.vocabSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.embedSize = embedSize;

            double scale = 1.0 / Math.sqrt(hiddenSize);
            this.embedding = initMatrix(vocabSize, embedSize, 0.5);

            this.W_i = initMatrix(hiddenSize, embedSize, scale); this.U_i = initMatrix(hiddenSize, hiddenSize, scale); this.b_i = new double[hiddenSize];
            this.W_f = initMatrix(hiddenSize, embedSize, scale); this.U_f = initMatrix(hiddenSize, hiddenSize, scale); this.b_f = initVector(hiddenSize, 1.0);
            this.W_o = initMatrix(hiddenSize, embedSize, scale); this.U_o = initMatrix(hiddenSize, hiddenSize, scale); this.b_o = new double[hiddenSize];
            this.W_g = initMatrix(hiddenSize, embedSize, scale); this.U_g = initMatrix(hiddenSize, hiddenSize, scale); this.b_g = new double[hiddenSize];

            this.W_out = initMatrix(vocabSize, hiddenSize, scale);
            this.b_out = new double[vocabSize];
        }

        public int[] autoregressiveDecode(DualLSTMState initState, int maxLen) {
            double[] h = Arrays.copyOf(initState.hiddenState(), hiddenSize);
            double[] c = Arrays.copyOf(initState.cellState(), hiddenSize);
            int currentToken = 0; // <GO>
            int[] generated = new int[maxLen];
            int count = 0;

            for (int step = 0; step < maxLen; step++) {
                double[] x_t = embedding[currentToken];
                double[] i_t = new double[hiddenSize];
                double[] f_t = new double[hiddenSize];
                double[] o_t = new double[hiddenSize];
                double[] g_t = new double[hiddenSize];

                for (int i = 0; i < hiddenSize; i++) {
                    double sum_i = b_i[i], sum_f = b_f[i], sum_o = b_o[i], sum_g = b_g[i];
                    for (int j = 0; j < embedSize; j++) {
                        sum_i += W_i[i][j] * x_t[j];
                        sum_f += W_f[i][j] * x_t[j];
                        sum_o += W_o[i][j] * x_t[j];
                        sum_g += W_g[i][j] * x_t[j];
                    }
                    for (int k = 0; k < hiddenSize; k++) {
                        sum_i += U_i[i][k] * h[k];
                        sum_f += U_f[i][k] * h[k];
                        sum_o += U_o[i][k] * h[k];
                        sum_g += U_g[i][k] * h[k];
                    }
                    i_t[i] = sigmoid(sum_i);
                    f_t[i] = sigmoid(sum_f);
                    o_t[i] = sigmoid(sum_o);
                    g_t[i] = Math.tanh(sum_g);
                }

                double[] next_c = new double[hiddenSize];
                double[] next_h = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    next_c[i] = (f_t[i] * c[i]) + (i_t[i] * g_t[i]);
                    next_h[i] = o_t[i] * Math.tanh(next_c[i]);
                }
                c = next_c;
                h = next_h;

                // Logits
                double maxLogit = Double.NEGATIVE_INFINITY;
                int bestToken = 0;
                for (int v = 0; v < vocabSize; v++) {
                    double sum = b_out[v];
                    for (int k = 0; k < hiddenSize; k++) sum += W_out[v][k] * h[k];
                    if (sum > maxLogit) {
                        maxLogit = sum;
                        bestToken = v;
                    }
                }
                generated[count++] = bestToken;
                if (bestToken == 4) break; // <EOS>
                currentToken = bestToken;
            }
            return Arrays.copyOf(generated, count);
        }
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

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" LECTURE 06: ENTERPRISE JAVA 17+ LSTM SEQ2SEQ TRANSLATOR");
        System.out.println(" Task: 'I go <EOS>' -> 'मैं जाता हूँ <EOS>'");
        System.out.println("================================================================================");

        int engVocabSize = 3;
        int hinVocabSize = 5;
        int embedSize = 2;
        int hiddenSize = 4;

        String[] hinWords = { "<GO>", "मैं", "जाता", "हूँ", "<EOS>" };

        TinyEncoderLSTM encoder = new TinyEncoderLSTM(engVocabSize, embedSize, hiddenSize);
        TinyDecoderLSTM decoder = new TinyDecoderLSTM(hinVocabSize, embedSize, hiddenSize);

        int[] srcTokens = { 0, 1, 2 }; // "I go <EOS>"
        DualLSTMState context = encoder.encode(srcTokens);

        System.out.println("Source Input Tokens: [0, 1, 2] ('I go <EOS>')");
        System.out.println("Encoded Hidden State (h): " + Arrays.toString(context.hiddenState()));
        System.out.println("Encoded Cell State (c)  : " + Arrays.toString(context.cellState()) + "  <── [WRITE-AHEAD LOG HIGHWAY]");

        int[] decoded = decoder.autoregressiveDecode(context, 6);
        System.out.print("\nDecoded Tokens: ");
        for (int tok : decoded) {
            System.out.print("'" + hinWords[tok] + "' ");
        }
        System.out.println("\n================================================================================");
    }
}
