"""
Lecture 4: Code an RNN from Scratch (PyTorch Implementation)
Task: English to Hindi Neural Machine Translation
Source: "I go <EOS>" -> Target: "मैं जाता हूँ <EOS>"

Demonstrates:
1. Manual TinyEncoder with explicit weight parameters (W_h, W_x, b).
2. Manual TinyDecoder with explicit weight parameters (W_h, W_x, b, W_out, b_out).
3. Teacher Forcing training loop with nn.CrossEntropyLoss and optim.SGD.
4. Autoregressive inference decoding.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------
# 1. Tiny Manual RNN Encoder
# ---------------------------------------------------------------------
class TinyEncoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)

        # RNN parameters
        self.hidden_size = hidden_size
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_x = nn.Parameter(torch.randn(hidden_size, embed_size) * 0.1)
        self.b   = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, src_tokens):
        """
        src_tokens: shape (src_len,)
        Returns final hidden state (hidden_size,).
        """
        h = torch.zeros(self.hidden_size)

        for t in range(src_tokens.shape[0]):
            token_id = src_tokens[t]
            x_t = self.embedding(token_id)

            # h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
            h = torch.tanh(
                torch.mv(self.W_h, h) +
                torch.mv(self.W_x, x_t) +
                self.b
            )

        return h


# ---------------------------------------------------------------------
# 2. Tiny Manual RNN Decoder
# ---------------------------------------------------------------------
class TinyDecoder(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_size)

        self.hidden_size = hidden_size
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_x = nn.Parameter(torch.randn(hidden_size, embed_size) * 0.1)
        self.b   = nn.Parameter(torch.zeros(hidden_size))

        # Output projection layer (hidden_size -> output_vocab_size)
        self.W_out = nn.Parameter(torch.randn(output_vocab_size, hidden_size) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(output_vocab_size))

    def forward(self, dec_tokens, init_hidden):
        h = init_hidden
        logits_list = []

        for t in range(dec_tokens.shape[0]):
            token_id = dec_tokens[t]
            x_t = self.embedding(token_id)

            # h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
            h = torch.tanh(
                torch.mv(self.W_h, h) +
                torch.mv(self.W_x, x_t) +
                self.b
            )
            # Logits: z_t = W_out @ h_t + b_out
            logits_t = torch.mv(self.W_out, h) + self.b_out
            logits_list.append(logits_t.unsqueeze(0))

        return torch.cat(logits_list, dim=0)


# ---------------------------------------------------------------------
# 3. Example Data & Training Configuration
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print(" LECTURE 04: TINY PYTORCH RNN TRANSLATOR")
    print("=" * 70)

    ENG_VOCAB_SIZE = 3  # I=0, go=1, <EOS>=2
    HIN_VOCAB_SIZE = 5  # <GO>=0, मैं=1, जाता=2, हूँ=3, <EOS>=4

    # Map IDs to words for printing
    HIN_ID2WORD = {
        0: "<GO>",
        1: "मैं",
        2: "जाता",
        3: "हूँ",
        4: "<EOS>"
    }

    EMBED_SIZE = 1
    HIDDEN_SIZE = 2

    encoder = TinyEncoder(ENG_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
    decoder = TinyDecoder(HIN_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)

    # Source: "I go <EOS>" => [0, 1, 2]
    encoder_input = torch.tensor([0, 1, 2])

    # Decoder target: "मैं जाता हूँ <EOS>" => [1, 2, 3, 4]
    # Teacher forcing in training:
    decoder_input  = torch.tensor([0, 1, 2, 3])  # <GO>, मैं, जाता, हूँ
    decoder_target = torch.tensor([1, 2, 3, 4])  #     मैं, जाता, हूँ, <EOS>

    # -----------------------------------------------------------------
    # 4. Training Loop (Cross Entropy + SGD)
    # -----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

    num_epochs = 1000
    print(f"Starting training for {num_epochs} epochs with SGD optimizer...\n")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 1) Encode source sentence -> context vector
        enc_hidden = encoder(encoder_input)  # shape (2,)

        # 2) Decode with teacher forcing
        logits = decoder(decoder_input, enc_hidden)  # shape (4, 5)

        # 3) Compute cross-entropy loss
        loss = criterion(logits, decoder_target)

        # 4) Backpropagation through time (BPTT) + parameter update
        loss.backward()
        optimizer.step()

        # Print stats
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{num_epochs}, Loss = {loss.item():.4f}")

        # -------------------------------------------------------------
        # Autoregressive Free-Running Inference (every 200 epochs)
        # -------------------------------------------------------------
        if (epoch + 1) % 200 == 0:
            print(f"\n--- Autoregressive Decoding after Epoch {epoch+1} ---")
            with torch.no_grad():
                # Re-encode source sentence
                enc_hidden = encoder(encoder_input)

                # Start with <GO> token (ID 0)
                current_token = torch.tensor(0)
                h = enc_hidden.clone()

                generated_tokens = []
                for _ in range(6):
                    x_t = decoder.embedding(current_token)
                    h = torch.tanh(
                        torch.mv(decoder.W_h, h) +
                        torch.mv(decoder.W_x, x_t) +
                        decoder.b
                    )

                    logits_t = torch.mv(decoder.W_out, h) + decoder.b_out
                    next_token = torch.argmax(logits_t).item()
                    generated_tokens.append(next_token)

                    if next_token == 4:  # <EOS>
                        break
                    current_token = torch.tensor(next_token)

                generated_words = [HIN_ID2WORD[t] for t in generated_tokens]
                print(f"Generated tokens: {generated_words}")
            print("----------------------------------------------------\n")
