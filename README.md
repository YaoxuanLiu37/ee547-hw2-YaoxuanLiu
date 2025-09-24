# EE547 HW2 

**Name:** YAOXUAN LIU  
**Email:** yaoxuanl@usc.edu  

# Problem 2

## Embedding Architecture

- **Input Representation**: Bag-of-Words (multi-hot vector) built from the top 5,000 most frequent tokens.  
- **Encoder**: Linear projection from vocabulary space → embedding dimension with ReLU activation.  
- **Bottleneck (Embedding Layer)**: Compact representation of size 256 (configurable).  
- **Decoder**: Linear projection from embedding → vocabulary space with tied weights, followed by Sigmoid activation.  
- **Loss Function**: Binary Cross-Entropy (BCE), treating reconstruction as a multi-label prediction task.

### Parameter Constraint
- Total parameters kept under **2,000,000** (actual ~137k with sample data).  
- Achieved by limiting vocabulary size and embedding dimension.

## Training & Outputs

During training:
- Input: tokenized and cleaned abstracts.  
- Training objective: minimize reconstruction loss of the Bag-of-Words vectors.  
- Loss decreases steadily across epochs, confirming proper learning.

Generated outputs:
1. **model.pth** – Trained PyTorch model with config and vocab.  
2. **embeddings.json** – Learned embeddings per paper with reconstruction loss.  
3. **vocabulary.json** – Token-to-index mapping and statistics.  
4. **training_log.json** – Training metadata (loss, parameters, epochs).  
