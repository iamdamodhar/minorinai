package com.vizuara.genai.tinyseq2seq;

import java.util.Arrays;

/**
 * Lecture 04: Tiny Seq2Seq Translator in Enterprise Java 17+.
 * Demonstrates:
 * 1. Manual Encoder-Decoder state transition.
 * 2. Teacher Forcing loss evaluation using Cross-Entropy.
 * 3. Autoregressive inference decoding.
 */
public class TinySeq2SeqTranslator {

    public static class TinyEncoder {
        private final double[][] embedding;
        private final double[][] W_h;
        private final double[][] W_x;
        private final double[] b;
        private final int hiddenSize;

        public TinyEncoder(int vocabSize, int embedSize, int hiddenSize) {
            this.hiddenSize = hiddenSize;
            this.embedding = initMatrix(vocabSize, embedSize, 0.5);
            this.W_h = initMatrix(hiddenSize, hiddenSize, 0.1);
            this.W_x = initMatrix(hiddenSize, embedSize, 0.1);
            this.b = new double[hiddenSize];
        }

        public double[] forward(int[] srcTokens) {
            double[] h = new double[hiddenSize];
            for (int token : srcTokens) {
                double[] x_t = embedding[token];
                double[] new_h = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    double sum = b[i];
                    for (int j = 0; j < hiddenSize; j++) sum += W_h[i][j] * h[j];
                    for (int k = 0; k < x_t.length; k++) sum += W_x[i][k] * x_t[k];
                    new_h[i] = Math.tanh(sum);
                }
                h = new_h;
            }
            return h;
        }
    }

    public static class TinyDecoder {
        private final double[][] embedding;
        private final double[][] W_h;
        private final double[][] W_x;
        private final double[] b;
        private final double[][] W_out;
        private final double[] b_out;
        private final int hiddenSize;
        private final int vocabSize;

        public TinyDecoder(int vocabSize, int embedSize, int hiddenSize) {
            this.vocabSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.embedding = initMatrix(vocabSize, embedSize, 0.5);
            this.W_h = initMatrix(hiddenSize, hiddenSize, 0.1);
            this.W_x = initMatrix(hiddenSize, embedSize, 0.1);
            this.b = new double[hiddenSize];
            this.W_out = initMatrix(vocabSize, hiddenSize, 0.1);
            this.b_out = new double[vocabSize];
        }

        public double[][] forwardTeacherForcing(int[] decTokens, double[] initHidden) {
            double[] h = Arrays.copyOf(initHidden, hiddenSize);
            double[][] allLogits = new double[decTokens.length][vocabSize];

            for (int t = 0; t < decTokens.length; t++) {
                double[] x_t = embedding[decTokens[t]];
                double[] new_h = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    double sum = b[i];
                    for (int j = 0; j < hiddenSize; j++) sum += W_h[i][j] * h[j];
                    for (int k = 0; k < x_t.length; k++) sum += W_x[i][k] * x_t[k];
                    new_h[i] = Math.tanh(sum);
                }
                h = new_h;

                for (int i = 0; i < vocabSize; i++) {
                    double sum = b_out[i];
                    for (int k = 0; k < hiddenSize; k++) sum += W_out[i][k] * h[k];
                    allLogits[t][i] = sum;
                }
            }
            return allLogits;
        }

        public int[] autoregressiveGenerate(double[] initHidden, int maxLen) {
            double[] h = Arrays.copyOf(initHidden, hiddenSize);
            int currentToken = 0; // <GO>
            int[] generated = new int[maxLen];
            int count = 0;

            for (int step = 0; step < maxLen; step++) {
                double[] x_t = embedding[currentToken];
                double[] new_h = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) {
                    double sum = b[i];
                    for (int j = 0; j < hiddenSize; j++) sum += W_h[i][j] * h[j];
                    for (int k = 0; k < x_t.length; k++) sum += W_x[i][k] * x_t[k];
                    new_h[i] = Math.tanh(sum);
                }
                h = new_h;

                double maxLogit = Double.NEGATIVE_INFINITY;
                int bestToken = 0;
                for (int i = 0; i < vocabSize; i++) {
                    double sum = b_out[i];
                    for (int k = 0; k < hiddenSize; k++) sum += W_out[i][k] * h[k];
                    if (sum > maxLogit) {
                        maxLogit = sum;
                        bestToken = i;
                    }
                }
                generated[count++] = bestToken;
                if (bestToken == 4) break; // <EOS>
                currentToken = bestToken;
            }
            return Arrays.copyOf(generated, count);
        }
    }

    public static double computeCrossEntropyLoss(double[][] logits, int[] targets) {
        double totalLoss = 0.0;
        for (int t = 0; t < targets.length; t++) {
            double[] logitRow = logits[t];
            double maxLogit = Arrays.stream(logitRow).max().orElse(0.0);
            double sumExp = Arrays.stream(logitRow).map(v -> Math.exp(v - maxLogit)).sum();
            double probTarget = Math.exp(logitRow[targets[t]] - maxLogit) / sumExp;
            totalLoss += -Math.log(Math.max(probTarget, 1e-12));
        }
        return totalLoss / targets.length;
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

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" LECTURE 04: ENTERPRISE JAVA 17+ TINY SEQ2SEQ TRANSLATOR");
        System.out.println("================================================================================");

        int engVocabSize = 3;
        int hinVocabSize = 5;
        int embedSize = 1;
        int hiddenSize = 2;

        String[] hinWords = { "<GO>", "मैं", "जाता", "हूँ", "<EOS>" };

        TinyEncoder encoder = new TinyEncoder(engVocabSize, embedSize, hiddenSize);
        TinyDecoder decoder = new TinyDecoder(hinVocabSize, embedSize, hiddenSize);

        int[] srcTokens = { 0, 1, 2 };       // "I go <EOS>"
        int[] decInput  = { 0, 1, 2, 3 };    // "<GO>, मैं, जाता, हूँ"
        int[] decTarget = { 1, 2, 3, 4 };    // "मैं, जाता, हूँ, <EOS>"

        // Forward Pass
        double[] encHidden = encoder.forward(srcTokens);
        double[][] logits = decoder.forwardTeacherForcing(decInput, encHidden);
        double loss = computeCrossEntropyLoss(logits, decTarget);

        System.out.println("\nSource Input: 'I go <EOS>' (Tokens: [0, 1, 2])");
        System.out.printf("Encoder Output Context Vector: [%.4f, %.4f]%n", encHidden[0], encHidden[1]);
        System.out.printf("Initial Teacher-Forcing Cross-Entropy Loss: %.4f%n", loss);

        int[] generated = decoder.autoregressiveGenerate(encHidden, 6);
        System.out.print("Untrained Autoregressive Generation: ");
        for (int tok : generated) System.out.print("'" + hinWords[tok] + "' ");
        System.out.println("\n================================================================================");
    }
}
