package com.vizuara.genai.gan;

import java.util.Arrays;
import java.util.Random;

/**
 * Generative Adversarial Networks (GAN) — Enterprise Java 17+ Clean Architecture.
 *
 * Maps Adversarial Training Mechanics to Staff Backend Systems Primitives:
 * - Latent Vector (z)                      : Compact PRNG Chaos Seed
 * - GeneratorService (G_theta)             : Synthetic Payload / Catalog Image Producer
 * - DiscriminatorService (D_phi)           : Adaptive Fraud / Authenticity Firewall
 * - inspectGradientReadOnly (Frozen D)     : Immutable Shadow-Mode Evaluation (Prevents the
 *                                            Keras 3 `d_model.trainable = False` aliasing bug!)
 * - Inverted Label Objective (y_gan = 1.0) : Non-Saturating Red-Team Optimization (-ln D(G(z)))
 */
public class AdversarialGANEngine {

    public record GeneratorForwardState(double[] z, double[] hiddenPre, double[] hiddenAct, double[] fakeSample) {}

    public record DiscriminatorForwardState(double[] input, double[] hiddenPre, double[] hiddenAct, double probability) {}

    public record AdversarialStepMetrics(double dLossReal, double dLossFake, double gLoss, double realProb, double fakeProb) {}

    // =========================================================================
    // ACTIVATION PHYSICS UTILITIES
    // =========================================================================
    public static final class Activations {
        private Activations() {}

        public static double leakyRelu(double x, double alpha) {
            return x >= 0.0 ? x : alpha * x;
        }

        public static double dLeakyRelu(double preAct, double alpha) {
            return preAct >= 0.0 ? 1.0 : alpha;
        }

        public static double sigmoid(double x) {
            double clamped = Math.max(-500.0, Math.min(500.0, x));
            return 1.0 / (1.0 + Math.exp(-clamped));
        }

        public static double binaryCrossEntropy(double prob, double label) {
            double eps = 1e-12;
            double p = Math.max(eps, Math.min(1.0 - eps, prob));
            return -(label * Math.log(p) + (1.0 - label) * Math.log(1.0 - p));
        }
    }

    // =========================================================================
    // GENERATOR SERVICE: z in R^{latentDim} -> LeakyReLU(0.2) -> tanh in [-1, 1]^{dataDim}
    // =========================================================================
    public static class GeneratorService {
        private final int latentDim;
        private final int hiddenDim;
        private final int outDim;
        private final double[][] W1;
        private final double[] b1;
        private final double[][] W2;
        private final double[] b2;

        public GeneratorService(int latentDim, int hiddenDim, int outDim, Random rng) {
            this.latentDim = latentDim;
            this.hiddenDim = hiddenDim;
            this.outDim = outDim;
            this.W1 = new double[hiddenDim][latentDim];
            this.b1 = new double[hiddenDim];
            this.W2 = new double[outDim][hiddenDim];
            this.b2 = new double[outDim];

            double scale1 = Math.sqrt(2.0 / latentDim) * 0.4;
            double scale2 = Math.sqrt(1.0 / hiddenDim) * 0.4;
            for (int i = 0; i < hiddenDim; i++) {
                for (int j = 0; j < latentDim; j++) {
                    W1[i][j] = rng.nextGaussian() * scale1;
                }
            }
            for (int k = 0; k < outDim; k++) {
                for (int i = 0; i < hiddenDim; i++) {
                    W2[k][i] = rng.nextGaussian() * scale2;
                }
            }
        }

        public GeneratorForwardState forward(double[] z) {
            double[] hPre = new double[hiddenDim];
            double[] hAct = new double[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) {
                double sum = b1[i];
                for (int j = 0; j < latentDim; j++) {
                    sum += W1[i][j] * z[j];
                }
                hPre[i] = sum;
                hAct[i] = Activations.leakyRelu(sum, 0.2);
            }

            double[] xFake = new double[outDim];
            for (int k = 0; k < outDim; k++) {
                double sum = b2[k];
                for (int i = 0; i < hiddenDim; i++) {
                    sum += W2[k][i] * hAct[i];
                }
                xFake[k] = Math.tanh(sum);
            }
            return new GeneratorForwardState(z, hPre, hAct, xFake);
        }

        public void applyInputGradients(GeneratorForwardState[] states, double[][] dLossDxFakeBatch, double lr) {
            int batchSize = states.length;
            double[][] gW1 = new double[hiddenDim][latentDim];
            double[] gb1 = new double[hiddenDim];
            double[][] gW2 = new double[outDim][hiddenDim];
            double[] gb2 = new double[outDim];

            for (int b = 0; b < batchSize; b++) {
                GeneratorForwardState st = states[b];
                double[] dLdx = dLossDxFakeBatch[b];

                // Backprop through tanh: d(tanh(u))/du = 1 - x_fake^2
                double[] dOutPre = new double[outDim];
                for (int k = 0; k < outDim; k++) {
                    double xf = st.fakeSample()[k];
                    dOutPre[k] = dLdx[k] * (1.0 - xf * xf);
                    gb2[k] += dOutPre[k];
                    for (int i = 0; i < hiddenDim; i++) {
                        gW2[k][i] += dOutPre[k] * st.hiddenAct()[i];
                    }
                }

                // Backprop through W2 and LeakyReLU(0.2)
                for (int i = 0; i < hiddenDim; i++) {
                    double dhAct = 0.0;
                    for (int k = 0; k < outDim; k++) {
                        dhAct += W2[k][i] * dOutPre[k];
                    }
                    double dhPre = dhAct * Activations.dLeakyRelu(st.hiddenPre()[i], 0.2);
                    gb1[i] += dhPre;
                    for (int j = 0; j < latentDim; j++) {
                        gW1[i][j] += dhPre * st.z()[j];
                    }
                }
            }

            for (int k = 0; k < outDim; k++) {
                b2[k] -= lr * (gb2[k] / batchSize);
                for (int i = 0; i < hiddenDim; i++) {
                    W2[k][i] -= lr * (gW2[k][i] / batchSize);
                }
            }
            for (int i = 0; i < hiddenDim; i++) {
                b1[i] -= lr * (gb1[i] / batchSize);
                for (int j = 0; j < latentDim; j++) {
                    W1[i][j] -= lr * (gW1[i][j] / batchSize);
                }
            }
        }
    }

    // =========================================================================
    // DISCRIMINATOR SERVICE: Explicit Separation of Mutating Train vs. Frozen Read-Only Pass
    // =========================================================================
    public static class DiscriminatorService {
        private final int inDim;
        private final int hiddenDim;
        private final double[][] W1;
        private final double[] b1;
        private final double[] w2;
        private double b2;

        public DiscriminatorService(int inDim, int hiddenDim, Random rng) {
            this.inDim = inDim;
            this.hiddenDim = hiddenDim;
            this.W1 = new double[hiddenDim][inDim];
            this.b1 = new double[hiddenDim];
            this.w2 = new double[hiddenDim];
            this.b2 = 0.0;

            double scale1 = Math.sqrt(2.0 / inDim) * 0.4;
            double scale2 = Math.sqrt(1.0 / hiddenDim) * 0.4;
            for (int i = 0; i < hiddenDim; i++) {
                w2[i] = rng.nextGaussian() * scale2;
                for (int j = 0; j < inDim; j++) {
                    W1[i][j] = rng.nextGaussian() * scale1;
                }
            }
        }

        public DiscriminatorForwardState forward(double[] x) {
            double[] hPre = new double[hiddenDim];
            double[] hAct = new double[hiddenDim];
            double logit = b2;

            for (int i = 0; i < hiddenDim; i++) {
                double sum = b1[i];
                for (int j = 0; j < inDim; j++) {
                    sum += W1[i][j] * x[j];
                }
                hPre[i] = sum;
                hAct[i] = Activations.leakyRelu(sum, 0.2);
                logit += w2[i] * hAct[i];
            }
            return new DiscriminatorForwardState(x, hPre, hAct, Activations.sigmoid(logit));
        }

        /** Step (a) & Step (b): Mutates ONLY Discriminator parameters phi. */
        public double trainOnBatch(double[][] batch, double targetLabel, double lr) {
            int n = batch.length;
            double[][] gW1 = new double[hiddenDim][inDim];
            double[] gb1 = new double[hiddenDim];
            double[] gw2 = new double[hiddenDim];
            double gb2 = 0.0;
            double totalLoss = 0.0;

            for (double[] x : batch) {
                DiscriminatorForwardState st = forward(x);
                totalLoss += Activations.binaryCrossEntropy(st.probability(), targetLabel);
                double dLogit = st.probability() - targetLabel;
                gb2 += dLogit;

                for (int i = 0; i < hiddenDim; i++) {
                    gw2[i] += dLogit * st.hiddenAct()[i];
                    double dhPre = (w2[i] * dLogit) * Activations.dLeakyRelu(st.hiddenPre()[i], 0.2);
                    gb1[i] += dhPre;
                    for (int j = 0; j < inDim; j++) {
                        gW1[i][j] += dhPre * x[j];
                    }
                }
            }

            b2 -= lr * (gb2 / n);
            for (int i = 0; i < hiddenDim; i++) {
                w2[i] -= lr * (gw2[i] / n);
                b1[i] -= lr * (gb1[i] / n);
                for (int j = 0; j < inDim; j++) {
                    W1[i][j] -= lr * (gW1[i][j] / n);
                }
            }
            return totalLoss / n;
        }

        /**
         * Step (c): Read-Only Shadow Mode (`d_model.trainable = False` + Inverted Label `y_gan = 1.0`).
         * Returns dL_G / dx_fake WITHOUT mutating any Discriminator weights!
         */
        public double[] inspectGradientReadOnly(double[] xFake, double[] lossHolder) {
            DiscriminatorForwardState st = forward(xFake);
            lossHolder[0] = Activations.binaryCrossEntropy(st.probability(), 1.0);
            double dLogit = st.probability() - 1.0; // Non-saturating gradient -(1 - D(G(z)))

            double[] dLdx = new double[inDim];
            for (int i = 0; i < hiddenDim; i++) {
                double dhPre = (w2[i] * dLogit) * Activations.dLeakyRelu(st.hiddenPre()[i], 0.2);
                for (int j = 0; j < inDim; j++) {
                    dLdx[j] += W1[i][j] * dhPre;
                }
            }
            return dLdx;
        }
    }

    private static double[] sampleLatent(int dim, Random rng) {
        double[] z = new double[dim];
        for (int d = 0; d < dim; d++) {
            z[d] = rng.nextGaussian();
        }
        return z;
    }

    private static double[][] sampleRealRetailBatch(int batchSize, Random rng) {
        double[][] batch = new double[batchSize][4];
        for (int b = 0; b < batchSize; b++) {
            double shift = rng.nextGaussian() * 0.08;
            double f0 = Math.max(-0.98, Math.min(0.98,  0.75 + shift + rng.nextGaussian() * 0.04));
            double f1 = Math.max(-0.98, Math.min(0.98,  0.80 - 0.5 * shift + rng.nextGaussian() * 0.04));
            double f2 = Math.max(-0.98, Math.min(0.98, -0.60 + shift + rng.nextGaussian() * 0.04));
            double f3 = Math.max(-0.98, Math.min(0.98,  0.80 * f0 + rng.nextGaussian() * 0.03));
            batch[b] = new double[] { f0, f1, f2, f3 };
        }
        return batch;
    }

    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println(" ENTERPRISE JAVA 17+ ADVERSARIAL GAN ENGINE (4D RETAIL APPAREL MANIFOLD)");
        System.out.println("================================================================================");

        Random rng = new Random(42L);
        int latentDim = 3;
        int dataDim = 4;
        GeneratorService generator = new GeneratorService(latentDim, 16, dataDim, rng);
        DiscriminatorService discriminator = new DiscriminatorService(dataDim, 16, rng);

        int halfBatch = 32;
        int fullBatch = 64;
        double lr = 0.045;

        for (int iter = 1; iter <= 1200; iter++) {
            // Step (a): Train Discriminator on Real Catalog Vectors (y = 1.0)
            double[][] realBatch = sampleRealRetailBatch(halfBatch, rng);
            double dLoss1 = discriminator.trainOnBatch(realBatch, 1.0, lr);

            // Step (b): Train Discriminator on Fake Generator Vectors (y = 0.0)
            double[][] fakeBatch = new double[halfBatch][dataDim];
            for (int b = 0; b < halfBatch; b++) {
                fakeBatch[b] = generator.forward(sampleLatent(latentDim, rng)).fakeSample();
            }
            double dLoss2 = discriminator.trainOnBatch(fakeBatch, 0.0, lr);

            // Step (c): Train Generator via Frozen Discriminator with Inverted Labels (y_gan = 1.0)
            GeneratorForwardState[] gStates = new GeneratorForwardState[fullBatch];
            double[][] dLdxBatch = new double[fullBatch][dataDim];
            double gLossSum = 0.0;
            double[] lossHolder = new double[1];

            for (int b = 0; b < fullBatch; b++) {
                gStates[b] = generator.forward(sampleLatent(latentDim, rng));
                dLdxBatch[b] = discriminator.inspectGradientReadOnly(gStates[b].fakeSample(), lossHolder);
                gLossSum += lossHolder[0];
            }
            generator.applyInputGradients(gStates, dLdxBatch, lr);

            if (iter == 1 || iter == 200 || iter == 1200) {
                double[] sampleFake = generator.forward(sampleLatent(latentDim, rng)).fakeSample();
                System.out.printf("Iter %4d | d1=%.3f, d2=%.3f, g=%.3f | Sample Fake=%s%n",
                    iter, dLoss1, dLoss2, (gLossSum / fullBatch),
                    Arrays.toString(Arrays.stream(sampleFake).map(v -> Math.round(v * 1000.0) / 1000.0).toArray()));
            }
        }
    }
}
