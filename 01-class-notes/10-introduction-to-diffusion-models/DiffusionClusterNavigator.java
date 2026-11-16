package com.vizuara.genai.diffusion;

import java.util.Arrays;

/**
 * Introduction to Diffusion Models (DDPM & Text-Conditioned Cluster Navigation) — Java 17+.
 *
 * Demonstrates:
 * 1. DDPMScheduler (`addNoise` forward jump & `step` reverse Markov denoising)
 * 2. High-Probability Distribution Clusters & Analytical Score Field (grad_{x_t} ln q_t(x_t))
 * 3. Multi-Word Text Embedding Cross-Attention Conditioning & Compositional Cluster Mixing
 */
public class DiffusionClusterNavigator {

    public record ScoreEstimate(double[] predictedNoise, double[] scoreVector, double[] clusterResponsibilities) {}

    public static class DDPMScheduler {
        private final int numTrainTimesteps;
        private final double[] betas;
        private final double[] alphas;
        private final double[] alphasCumprod;

        public DDPMScheduler(int numTrainTimesteps, double betaStart, double betaEnd) {
            this.numTrainTimesteps = numTrainTimesteps;
            this.betas = new double[numTrainTimesteps];
            this.alphas = new double[numTrainTimesteps];
            this.alphasCumprod = new double[numTrainTimesteps];

            double cum = 1.0;
            for (int t = 0; t < numTrainTimesteps; t++) {
                betas[t] = betaStart + (betaEnd - betaStart) * ((double) t / (numTrainTimesteps - 1));
                alphas[t] = 1.0 - betas[t];
                cum *= alphas[t];
                alphasCumprod[t] = cum;
            }
        }

        /** Closed-form forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise */
        public double[] addNoise(double[] x0, double[] noise, int t) {
            double ab = alphasCumprod[t];
            double sqrtAb = Math.sqrt(ab);
            double sqrtOneMinusAb = Math.sqrt(1.0 - ab);
            double[] xt = new double[x0.length];
            for (int d = 0; d < x0.length; d++) {
                xt[d] = sqrtAb * x0[d] + sqrtOneMinusAb * noise[d];
            }
            return xt;
        }

        /** Reverse DDPM step: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1-alpha_t)/sqrt(1-alpha_bar_t)) * eps) */
        public double[] step(double[] modelPredNoise, int t, double[] xt) {
            double alphaT = alphas[t];
            double abT = alphasCumprod[t];
            double coeff = (1.0 - alphaT) / Math.sqrt(1.0 - abT);
            double invSqrtAlpha = 1.0 / Math.sqrt(alphaT);

            double[] prevSample = new double[xt.length];
            for (int d = 0; d < xt.length; d++) {
                prevSample[d] = invSqrtAlpha * (xt[d] - coeff * modelPredNoise[d]);
            }
            return prevSample;
        }

        public double alphaCumprod(int t) {
            return alphasCumprod[t];
        }
    }

    public static ScoreEstimate computeClusterScoreAndNoise(
        double[] xt,
        int t,
        DDPMScheduler scheduler,
        double[][] clusterCenters,
        double clusterSigma
    ) {
        double ab = scheduler.alphaCumprod(t);
        double sqrtAb = Math.sqrt(ab);
        double varT = ab * (clusterSigma * clusterSigma) + (1.0 - ab);

        int kCount = clusterCenters.length;
        double[] logWeights = new double[kCount];
        double maxLw = Double.NEGATIVE_INFINITY;

        for (int k = 0; k < kCount; k++) {
            double distSq = 0.0;
            for (int d = 0; d < xt.length; d++) {
                double diff = xt[d] - sqrtAb * clusterCenters[k][d];
                distSq += diff * diff;
            }
            logWeights[k] = -0.5 * distSq / varT;
            if (logWeights[k] > maxLw) {
                maxLw = logWeights[k];
            }
        }

        double[] resp = new double[kCount];
        double sumExp = 0.0;
        for (int k = 0; k < kCount; k++) {
            resp[k] = Math.exp(logWeights[k] - maxLw);
            sumExp += resp[k];
        }
        for (int k = 0; k < kCount; k++) {
            resp[k] /= sumExp;
        }

        double[] x0Hat = new double[xt.length];
        for (int k = 0; k < kCount; k++) {
            for (int d = 0; d < xt.length; d++) {
                x0Hat[d] += resp[k] * clusterCenters[k][d];
            }
        }

        double sqrtOneMinusAb = Math.max(Math.sqrt(1.0 - ab), 1e-8);
        double[] predNoise = new double[xt.length];
        double[] score = new double[xt.length];
        for (int d = 0; d < xt.length; d++) {
            predNoise[d] = (xt[d] - sqrtAb * x0Hat[d]) / sqrtOneMinusAb;
            score[d] = -predNoise[d] / sqrtOneMinusAb;
        }
        return new ScoreEstimate(predNoise, score, resp);
    }

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" ENTERPRISE JAVA 17+ DIFFUSION CLUSTER NAVIGATOR & COMPOSITIONAL MIXER");
        System.out.println("================================================================================");

        DDPMScheduler scheduler = new DDPMScheduler(50, 0.001, 0.12);
        double[][] clusters = {
            { -0.80, +0.80 }, // Cluster A: Blue Butterfly
            { +0.80, +0.80 }, // Cluster B: Golden Butterfly
            { -0.80, -0.80 }  // Cluster C: Blue Lotus
        };

        // Compositional Target for "golden lotus": Color=+0.80 (from Cluster B), Structure=-0.80 (from Cluster C)
        double[] composedTarget = { +0.80, -0.80 };
        double[] x = { -0.20, +0.30 };

        for (int t = 49; t >= 0; t--) {
            double ab = scheduler.alphaCumprod(t);
            double sqrtAb = Math.sqrt(ab);
            double sqrtOneMinusAb = Math.max(Math.sqrt(1.0 - ab), 1e-8);
            double[] epsCond = {
                (x[0] - sqrtAb * composedTarget[0]) / sqrtOneMinusAb,
                (x[1] - sqrtAb * composedTarget[1]) / sqrtOneMinusAb
            };
            x = scheduler.step(epsCond, t, x);
            if (t == 49 || t == 20 || t == 0) {
                System.out.printf("Reverse Step t=%2d | Trajectory x_t = %s%n",
                    t, Arrays.toString(Arrays.stream(x).map(v -> Math.round(v * 1000.0) / 1000.0).toArray()));
            }
        }
    }
}
