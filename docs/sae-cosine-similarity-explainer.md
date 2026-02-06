Great questions. Let me explain the cosine
  similarity measurement and then update the
  documentation.

  What Cosine Similarity Measures

  It's comparing vectors in activation space,
  not parameter space. Here's the flow:

  Input tokens → Model forward pass → Layer L
  produces activations
                                            ↓
                              [batch, seq_len,
  d_model]
                              e.g., [1, 50,
  2304] for Gemma 2 2B

  The roundtrip test does:
  original_activations (shape: [1, 50, 2304])
          ↓
     SAE.encode() → sparse features (shape: [1, 
  50, 16384])
          ↓
     SAE.decode() → reconstructed_activations
  (shape: [1, 50, 2304])
          ↓
     Compare: cosine_similarity(original,
  reconstructed)

  So we're comparing:
  - original: The actual hidden state vector at
  layer L (2304-dimensional for Gemma 2 2B)
  - reconstructed: What we get after encoding to
   sparse features and decoding back

  The SAE is a learned compression: it projects
  the 2304-dim activation into a 16384-dim
  sparse space (where most entries are ~0), then
   projects back. If reconstruction is perfect,
  cos_sim = 1.0.

  Why this matters for steering: When we steer,
  we do encode → modify one feature → decode. If
   the roundtrip alone introduces error, our
  modification is confounded by that noise.

  L2 Error Relationship

  - Cosine similarity = directional alignment
  (are vectors pointing same way?)
  - L2 error = magnitude of difference (how far
  apart are the vectors?)

  They're related but not redundant:
  cos_sim = (original · reconstructed) / 
  (||original|| × ||reconstructed||)
  L2_error = ||original - reconstructed||
  relative_L2 = L2_error / ||original|| × 100%

  You can have high cos_sim but high L2 if the
  vectors point the same direction but have
  different magnitudes.

  Let me update RESULTS_AND_METHODOLOGY.md with
  this explanation and think about what we want
  for n>1: