package com.vizuara.genai.lstm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Lecture 05: LSTM Network from Scratch in Enterprise Java 17+.
 * 
 * Maps Neural Network Memory Gates to Distributed Systems Primitives:
 * - Cell State (C_t): Append-Only Write-Ahead Log (WAL)
 * - Forget Gate (f_t): TTL / Topic Compaction Eviction Filter
 * - Input Gate (i_t * c_tilde): Ingestion Filter & Event Appender
 * - Output Gate (o_t): CQRS Read-Model Projection
 */
public class LSTMFromScratch {

    // Immutable State Container
    public record LSTMState(double[] cellState, double[] hiddenState) {}

    // Diagnostic Gate Snapshot
    public record GateSnapshot(
        double[] forgetGate,
        double[] inputGate,
        double[] candidateState,
        double[] cellState,
        double[] outputGate,
        double[] hiddenState
    ) {}

    public static class LSTMCell {
        private final int inputDim;
        private final int hiddenDim;

        // 1. Forget Gate Weights
        private final double[][] W_f;
        private final double[][] U_f;
        private final double[] b_f;

        // 2. Input Gate Weights
        private final double[][] W_i;
        private final double[][] U_i;
        private final double[] b_i;

        // 3. Candidate State Weights
        private final double[][] W_c;
        private final double[][] U_c;
        private final double[] b_c;

        // 4. Output Gate Weights
        private final double[][] W_o;
        private final double[][] U_o;
        private final double[] b_o;

        public LSTMCell(int inputDim, int hiddenDim) {
            this.inputDim = inputDim;
            this.hiddenDim = hiddenDim;

            double scale = 1.0 / Math.sqrt(hiddenDim);

            this.W_f = initMatrix(hiddenDim, inputDim, scale);
            this.U_f = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_f = initVector(hiddenDim, 1.0); // Forget bias initialized to 1.0

            this.W_i = initMatrix(hiddenDim, inputDim, scale);
            this.U_i = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_i = initVector(hiddenDim, 0.0);

            this.W_c = initMatrix(hiddenDim, inputDim, scale);
            this.U_c = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_c = initVector(hiddenDim, 0.0);

            this.W_o = initMatrix(hiddenDim, inputDim, scale);
            this.U_o = initMatrix(hiddenDim, hiddenDim, scale);
            this.b_o = initVector(hiddenDim, 0.0);
        }

        public GateSnapshot step(double[] x_t, LSTMState prevState) {
            double[] h_prev = prevState.hiddenState();
            double[] C_prev = prevState.cellState();

            // 1. Forget Gate: f_t = sigma(W_f @ x_t + U_f @ h_prev + b_f)
            double[] f_t = computeGate(W_f, x_t, U_f, h_prev, b_f, LSTMCell::sigmoid);

            // 2. Input Gate: i_t = sigma(W_i @ x_t + U_i @ h_prev + b_i)
            double[] i_t = computeGate(W_i, x_t, U_i, h_prev, b_i, LSTMCell::sigmoid);

            // 3. Candidate Memory: C_tilde = tanh(W_c @ x_t + U_c @ h_prev + b_c)
            double[] c_tilde = computeGate(W_c, x_t, U_c, h_prev, b_c, Math::tanh);

            // 4. Write-Ahead Log (WAL) Additive Update: C_t = (f_t * C_prev) + (i_t * c_tilde)
            double[] C_t = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                C_t[i] = (f_t[i] * C_prev[i]) + (i_t[i] * c_tilde[i]);
            }

            // 5. Output Gate: o_t = sigma(W_o @ x_t + U_o @ h_prev + b_o)
            double[] o_t = computeGate(W_o, x_t, U_o, h_prev, b_o, LSTMCell::sigmoid);

            // 6. CQRS Materialized View Projection: h_t = o_t * tanh(C_t)
            double[] h_t = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                h_t[i] = o_t[i] * Math.tanh(C_t[i]);
            }

            return new GateSnapshot(f_t, i_t, c_tilde, C_t, o_t, h_t);
        }

        private static double[] computeGate(double[][] W, double[] x, double[][] U, double[] h, double[] b, DoubleUnaryOperator act) {
            double[] result = new double[b.length];
            for (int i = 0; i < b.length; i++) {
                double sum = b[i];
                for (int j = 0; j < x.length; j++) sum += W[i][j] * x[j];
                for (int k = 0; k < h.length; k++) sum += U[i][k] * h[k];
                result[i] = act.apply(sum);
            }
            return result;
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

        @FunctionalInterface
        private interface DoubleUnaryOperator {
            double apply(double x);
        }
    }

    // =================================================================
    // MAIN DEMO & DOMAIN FEATURE TRACE
    // =================================================================
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" LECTURE 05: ENTERPRISE JAVA 17+ LSTM IMPLEMENTATION");
        System.out.println("================================================================================");

        int inputDim = 4;
        int hiddenDim = 4;

        LSTMCell lstm = new LSTMCell(inputDim, hiddenDim);

        // Retail Domain Entity Vector: [Formality, Warmth, PriceTier, Category]
        Map<String, double[]> catalogEmbeddings = new HashMap<>();
        catalogEmbeddings.put("WinterJacket", new double[]{ -0.2, 0.95, 0.8, 0.1 });
        catalogEmbeddings.put("WoolenScarf",  new double[]{ -0.4, 0.90, 0.3, -0.9 });
        catalogEmbeddings.put("ThermalBoots", new double[]{ -0.5, 0.85, 0.7, 1.0 });

        String[] sessionStream = { "WinterJacket", "WoolenScarf", "ThermalBoots" };

        LSTMState state = new LSTMState(new double[hiddenDim], new double[hiddenDim]);

        System.out.println("\nProcessing Retail User Clickstream Sequence:\n");

        for (int t = 0; t < sessionStream.length; t++) {
            String item = sessionStream[t];
            double[] x_t = catalogEmbeddings.get(item);

            GateSnapshot snapshot = lstm.step(x_t, state);
            state = new LSTMState(snapshot.cellState(), snapshot.hiddenState());

            System.out.printf("Step %d | Ingested SKU: %-15s%n", (t + 1), item);
            System.out.printf("  Forget Gate (f_t)  : %s  (Cache Eviction Ratio)%n", formatVec(snapshot.forgetGate()));
            System.out.printf("  Input Gate (i_t)   : %s  (Event Ingestion Filter)%n", formatVec(snapshot.inputGate()));
            System.out.printf("  Cell State (C_t)   : %s  <── [WRITE-AHEAD LOG HIGHWAY]%n", formatVec(snapshot.cellState()));
            System.out.printf("  Hidden State (h_t) : %s  <── [CQRS READ-MODEL PROJECTION]%n", formatVec(snapshot.hiddenState()));
            System.out.println("--------------------------------------------------------------------------------");
        }

        System.out.println("\nEnterprise Takeaway: LSTM Cell State functions identically to an event-sourced");
        System.out.println("Append-Only WAL in Kafka/Databases, preserving customer context across time.");
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
