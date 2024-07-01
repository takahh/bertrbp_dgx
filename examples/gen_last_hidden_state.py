# -------------------------------------------------------------------
# this code generates last hidden state from pretrained dnabert
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from transformers import BertTokenizer, BertModel
import torch
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():

    # Load the pretrained DNABERT model and tokenizer
    model_name = "zhihan1996/DNA_bert_6"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Example DNA sequence (must be tokenized appropriately)
    dna_sequence = "ACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    kmer_length = 6  # For DNABERT-6

    # Tokenize the input sequence
    def tokenize_dna_sequence(sequence, kmer_length):
        kmers = [sequence[i:i + kmer_length] for i in range(len(sequence) - kmer_length + 1)]
        return " ".join(kmers)

    tokenized_sequence = tokenize_dna_sequence(dna_sequence, kmer_length)
    inputs = tokenizer(tokenized_sequence, return_tensors="pt")

    # Get the outputs from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # The last hidden state
    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state.shape)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()