"""
Visualization utilities for the NMT project.
Uses matplotlib and seaborn to:
- Plot training loss & BLEU score over epochs.
- Plot attention heatmap for each translated sentence.
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history_path="checkpoints/train_history.json",
                          save_path=None, show=True):
    """
    Plot training loss and BLEU score over epochs.
    
    Args:
        history_path: Path to train_history.json file
        save_path: If not None, save the image to file
        show: If True, display the plot
    
    Returns:
        fig: matplotlib Figure object
    """
    if not os.path.exists(history_path):
        print(f"[!] File not found: {history_path}")
        return None

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if len(history) == 0:
        print("[!] No training history data.")
        return None

    epochs = [entry["epoch"] for entry in history]
    losses = [entry["avg_loss"] for entry in history]
    bleu_scores = [entry.get("bleu", None) for entry in history]
    has_bleu = any(b is not None for b in bleu_scores)

    # Set up style
    sns.set_theme(style="whitegrid", palette="muted")

    if has_bleu:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # ===== Plot Loss =====
    ax1.plot(epochs, losses, marker='o', color='#e74c3c', linewidth=2,
             markersize=6, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Mark the epoch with the best loss
    best_loss_idx = losses.index(min(losses))
    ax1.annotate(
        f'Best: {losses[best_loss_idx]:.4f}',
        xy=(epochs[best_loss_idx], losses[best_loss_idx]),
        xytext=(10, 15), textcoords='offset points',
        fontsize=10, color='#e74c3c',
        arrowprops=dict(arrowstyle='->', color='#e74c3c'),
    )

    # ===== Plot BLEU =====
    if has_bleu:
        valid_epochs = [e for e, b in zip(epochs, bleu_scores) if b is not None]
        valid_bleu = [b for b in bleu_scores if b is not None]

        ax2.plot(valid_epochs, valid_bleu, marker='s', color='#2ecc71',
                 linewidth=2, markersize=6, label='BLEU Score')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('BLEU Score', fontsize=12)
        ax2.set_title('BLEU Score', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Mark the epoch with the best BLEU
        if valid_bleu:
            best_bleu_idx = valid_bleu.index(max(valid_bleu))
            ax2.annotate(
                f'Best: {valid_bleu[best_bleu_idx]:.4f}',
                xy=(valid_epochs[best_bleu_idx], valid_bleu[best_bleu_idx]),
                xytext=(10, -15), textcoords='offset points',
                fontsize=10, color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71'),
            )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Saved training history plot: {save_path}")

    if show:
        plt.show()

    return fig


def plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens,
                           save_path=None, show=True):
    """
    Plot heatmap of attention weights between source and target sentences.
    
    Args:
        attention_weights: tensor/array shape (tgt_len, src_len)
        src_tokens: list of source tokens (string)
        tgt_tokens: list of target tokens (string)
        save_path: If not None, save the image to file
        show: If True, display the plot
    
    Returns:
        fig: matplotlib Figure object
    """
    import torch

    if attention_weights is None:
        print("[!] No attention weights to display.")
        return None

    # Convert tensor to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attn_data = attention_weights.detach().cpu().numpy()
    else:
        attn_data = attention_weights

    # Trim attention to match number of tokens
    tgt_len = min(len(tgt_tokens), attn_data.shape[0])
    src_len = min(len(src_tokens), attn_data.shape[1])
    attn_data = attn_data[:tgt_len, :src_len]

    # Set flexible figure size
    fig_width = max(6, src_len * 0.6 + 2)
    fig_height = max(4, tgt_len * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        attn_data,
        xticklabels=src_tokens[:src_len],
        yticklabels=tgt_tokens[:tgt_len],
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
    )

    ax.set_xlabel('Source Tokens (English)', fontsize=12)
    ax.set_ylabel('Target Tokens (Vietnamese)', fontsize=12)
    ax.set_title('Attention Heatmap', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Saved attention heatmap: {save_path}")

    if show:
        plt.show()

    return fig


def plot_translation_with_attention(evaluator, sentence, max_len=50,
                                    save_path=None, show=True):
    """
    Translate a sentence and plot the attention heatmap simultaneously.
    
    Args:
        evaluator: Evaluator object (with model, src_vocab, tgt_vocab)
        sentence: English sentence to translate
        max_len: Maximum length
        save_path: Path to save the figure (if needed)
        show: Whether to display the figure
    
    Returns:
        translated_words: list of translated tokens
        fig: matplotlib Figure
    """
    translated_words, attn_weights = evaluator.translate_sentence(
        sentence, max_len=max_len
    )

    # Get source tokens
    src_tokens = evaluator.src_vocab.tokenizer.tokenize(sentence)
    src_tokens = ['<SOS>'] + src_tokens + ['<EOS>']

    # Target tokens
    tgt_tokens = translated_words

    if len(tgt_tokens) == 0:
        print(f"[!] Could not translate sentence: '{sentence}'")
        return translated_words, None

    fig = plot_attention_heatmap(
        attn_weights,
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        save_path=save_path,
        show=show,
    )

    print(f"[SRC]  {sentence}")
    print(f"[PRED] {' '.join(translated_words)}")

    return translated_words, fig
