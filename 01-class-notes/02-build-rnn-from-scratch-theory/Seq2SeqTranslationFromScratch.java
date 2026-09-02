package com.vizuara.genai.seq2seq;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Module 02: Seq2Seq English-to-Hindi Translation in Enterprise Java 17+.
 * Demonstrates:
 * 1. Explicit Encoder-Decoder matrix computations from scratch.
 * 2. Context Vector compression hand-off (c = h_enc_T).
 * 3. Autoregressive Hindi token generation using Softmax and Temperature scaling.
 */
public class Seq2SeqTranslationFromScratch {

    public record DecodeOutput(double[] nextState, double[] logits, double[] probabilities, int argmaxIdx) {}

    public static class Seq2SeqModel {
        private final int embDim;
        private final int hiddenDim;
        private final int tgtVocabSize;

        // Encoder weights
        private final double[][] W_xh_enc;
        private final double[][] W_hh_enc;
        private final double[] b_h_enc;

        // Decoder weights
        private final double[][] W_ys_dec;
        private final double[][] W_ss_dec;
        private final double[] b_s_dec;
        private final double[][] W_vocab;
        private final double[] b_vocab;

        public Seq2SeqModel(int embDim, int hiddenDim, int tgtVocabSize) {
            this.embDim = embDim;
            this.hiddenDim = hiddenDim;
            this.tgtVocabSize = tgtVocabSize;

            double scale = 1.0 / Math.sqrt(hiddenDim);

            this.W_xh_enc = initMatrix(hiddenDim, embDim, scale);
            this.W_hh_enc = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_h_enc = new double[hiddenDim];

            this.W_ys_dec = initMatrix(hiddenDim, embDim, scale);
            this.W_ss_dec = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_s_dec = new double[hiddenDim];

            this.W_vocab = initMatrix(tgtVocabSize, hiddenDim, scale);
            this.b_vocab = new double[tgtVocabSize];
        }

        public double[] encode(double[][] sourceEmbeddings) {
            double[] h = new double[hiddenDim]; // h_0 = zeros
            for (double[] x_t : sourceEmbeddings) {
                double[] new_h = new double[hiddenDim];
                for (int i = 0; i < hiddenDim; i++) {
                    double sum = b_h_enc[i];
                    for (int j = 0; j < embDim; j++) sum += W_xh_enc[i][j] * x_t[j];
                    for (int k = 0; k < hiddenDim; k++) sum += W_hh_enc[i][k] * h[k];
                    new_h[i] = Math.tanh(sum);
                }
                h = new_h;
            }
            return h; // Returns final Context Vector c
        }

        public DecodeOutput decodeStep(double[] y_prev_emb, double[] s_prev, double temperature) {
            // 1. Decoder state update: s_t = tanh(W_ys @ y_{t-1} + W_ss @ s_{t-1} + b_s)
            double[] s_t = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                double sum = b_s_dec[i];
                for (int j = 0; j < embDim; j++) sum += W_ys_dec[i][j] * y_prev_emb[j];
                for (int k = 0; k < hiddenDim; k++) sum += W_ss_dec[i][k] * s_prev[k];
                s_t[i] = Math.tanh(sum);
            }

            // 2. Logits projection: z_t = W_vocab @ s_t + b_vocab
            double[] logits = new double[tgtVocabSize];
            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < tgtVocabSize; i++) {
                double sum = b_vocab[i];
                for (int k = 0; k < hiddenDim; k++) sum += W_vocab[i][k] * s_t[k];
                logits[i] = sum;
                double scaled = sum / temperature;
                if (scaled > maxLogit) maxLogit = scaled;
            }

            // 3. Numerically stable Softmax with Temperature
            double sumExp = 0.0;
            double[] probs = new double[tgtVocabSize];
            for (int i = 0; i < tgtVocabSize; i++) {
                probs[i] = Math.exp((logits[i] / temperature) - maxLogit);
                sumExp += probs[i];
            }
            int argmaxIdx = 0;
            double maxProb = -1.0;
            for (int i = 0; i < tgtVocabSize; i++) {
                probs[i] /= sumExp;
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    argmaxIdx = i;
                }
            }

            return new DecodeOutput(s_t, logits, probs, argmaxIdx);
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
        System.out.println(" MODULE 02: ENTERPRISE JAVA 17+ SEQ2SEQ ENGLISH-TO-HINDI TRANSLATOR");
        System.out.println("================================================================================");

        int embDim = 4;
        int hiddenDim = 3;
        int tgtVocabSize = 6;

        Map<String, double[]> srcEmb = new HashMap<>();
        srcEmb.put("You",  new double[]{ 0.8, -0.3,  0.9, -0.1 });
        srcEmb.put("are",  new double[]{-0.4,  0.7, -0.1,  0.6 });
        srcEmb.put("good", new double[]{ 0.1,  0.9,  0.8,  0.4 });

        Map<String, double[]> tgtEmb = new HashMap<>();
        tgtEmb.put("<SOS>", new double[]{ 0.1,  0.1,  0.1,  0.1 });
        tgtEmb.put("आप",    new double[]{ 0.9, -0.2,  0.8, -0.1 });
        tgtEmb.put("अच्छे", new double[]{ 0.2,  0.8,  0.7,  0.3 });
        tgtEmb.put("हैं",   new double[]{-0.3,  0.6, -0.2,  0.5 });
        tgtEmb.put("<EOS>", new double[]{-0.9, -0.1,  0.0, -0.8 });

        String[] idx2tgt = { "<PAD>", "<SOS>", "आप", "अच्छे", "हैं", "<EOS>" };

        Seq2SeqModel model = new Seq2SeqModel(embDim, hiddenDim, tgtVocabSize);

        // 1. Encode "You are good"
        double[][] srcSeq = { srcEmb.get("You"), srcEmb.get("are"), srcEmb.get("good") };
        double[] contextVector = model.encode(srcSeq);

        System.out.println("\nSource English Sentence: 'You are good'");
        System.out.println("Generated Context Vector c (size " + hiddenDim + "): " + formatVec(contextVector));

        // 2. Decode Autoregressively
        System.out.println("\nAutoregressive Decoding of Hindi Translation:\n");
        double[] s_curr = contextVector;
        String currentToken = "<SOS>";

        for (int t = 1; t <= 4; t++) {
            double[] y_emb = tgtEmb.getOrDefault(currentToken, tgtEmb.get("<SOS>"));
            DecodeOutput out = model.decodeStep(y_emb, s_curr, 1.0);
            s_curr = out.nextState();
            String predicted = idx2tgt[out.argmaxIdx()];

            System.out.printf("Step %d | Input: '%-5s' ──► State s_%d: %s ──► Argmax: '%-5s' (p=%.3f)%n",
                t, currentToken, t, formatVec(s_curr), predicted, out.probabilities()[out.argmaxIdx()]);

            currentToken = predicted;
        }

        System.out.println("\n================================================================================");
        System.out.println(" Enterprise Takeaway: The 3-number Context Vector bottleneck forced the invention");
        System.out.println(" of Bahdanau Attention (2014) and 2017 Transformers (Attention Is All You Need).");
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
